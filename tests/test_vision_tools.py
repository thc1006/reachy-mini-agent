"""Track C TDD: vision tools powered by Qwen3.6 native vision (MMMU 81.7, RefCOCO 92).

Three new tools (all VL-only, no motor — motor API investigation is v2):
  see_what(query?)          — caption current frame; optional query narrows focus
  find_in_view(description) — ground bbox + relative position for an object
  count_items(description)  — count occurrences of an object class in frame

Tests here are pure-logic / subprocess-isolated so we don't drag robot_brain's
full Whisper + Mediapipe stack into memory.
"""
import base64
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY   = str(ROOT / ".venv" / "bin" / "python")


def _make_scene_jpeg_b64() -> str:
    """Generate a test scene with known content for deterministic assertions."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (640, 480), color=(235, 235, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 300, 180, 450], fill=(220, 30, 30))       # red square, lower-left
    d.ellipse([440, 60, 600, 220], fill=(30, 80, 220))          # blue circle, upper-right
    d.ellipse([270, 200, 370, 300], fill=(30, 180, 50))         # green circle, center
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        d.text((250, 70), "HELLO", fill=(10, 10, 10), font=font)
    except Exception:
        d.text((250, 70), "HELLO", fill=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _run(script: str, env_extra=None, timeout=180):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra: env.update(env_extra)
    return subprocess.run([PY, "-c", script], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)


# --- Schema + registry tests (no LLM) ---------------------------------------

class TestVisionToolRegistry:
    def test_new_tools_registered(self):
        from robot_tools import TOOLS
        for name in ("see_what", "find_in_view", "count_items"):
            assert name in TOOLS, f"missing {name}"

    def test_specs_json_serializable(self):
        import json
        from robot_tools import TOOLS
        for name in ("see_what", "find_in_view", "count_items"):
            spec, _ = TOOLS[name]
            json.dumps(spec)


# --- No-frame path: graceful error, no crash ---------------------------------

class TestNoFrame:
    def test_see_what_returns_error_when_no_frame(self):
        r = _run(
            "import robot_tools as rt; "
            "import robot_tools as m; "
            "m._get_frame_b64 = lambda: None; "
            "r = rt.TOOLS['see_what'][1](); "
            "print(f'ERR={\"error\" in r}')"
        )
        assert "ERR=True" in r.stdout, f"{r.stderr}"

    def test_find_in_view_returns_error_when_no_frame(self):
        r = _run(
            "import robot_tools as rt, robot_tools as m; "
            "m._get_frame_b64 = lambda: None; "
            "r = rt.TOOLS['find_in_view'][1](description='cup'); "
            "print(f'ERR={\"error\" in r}')"
        )
        assert "ERR=True" in r.stdout, f"{r.stderr}"


# --- End-to-end with qwen3.6 vision + test JPEG (integration) ---------------

class TestVisionE2E:
    """These hit the Ollama qwen3.6:35b-a3b VL endpoint for real. Cold load ~7s
    first time, then <1s per call."""

    def test_see_what_describes_scene(self):
        b64 = _make_scene_jpeg_b64()
        script = f"""
import robot_tools as rt, robot_tools as m
m._get_frame_b64 = lambda: {b64!r}
r = rt.TOOLS['see_what'][1]()
print(f'OK={{\"description\" in r and len(r[\"description\"]) > 10}}')
print(f'DESC={{r.get(\"description\",\"\")[:200]!r}}')
"""
        r = _run(script, timeout=120)
        assert "OK=True" in r.stdout, f"{r.stderr[-400:]}"

    def test_see_what_with_query_focuses(self):
        b64 = _make_scene_jpeg_b64()
        script = f"""
import robot_tools as rt, robot_tools as m
m._get_frame_b64 = lambda: {b64!r}
r = rt.TOOLS['see_what'][1](query='What color is the circle in the upper right?')
print(f'DESC={{r.get(\"description\",\"\")[:200].lower()!r}}')
print(f'HAS_BLUE={{\"blue\" in r.get(\"description\",\"\").lower()}}')
"""
        r = _run(script, timeout=120)
        assert "HAS_BLUE=True" in r.stdout, f"{r.stderr[-400:]}"

    def test_find_in_view_returns_bbox_for_known_object(self):
        b64 = _make_scene_jpeg_b64()
        script = f"""
import robot_tools as rt, robot_tools as m
m._get_frame_b64 = lambda: {b64!r}
r = rt.TOOLS['find_in_view'][1](description='the red square')
print(f'FOUND={{r.get(\"found\")}}')
bbox = r.get('bbox')
if bbox and len(bbox) == 4:
    x, y, w, h = bbox
    # our test scene has the red square roughly at x=40-180, y=300-450
    # acceptance is lenient — just sanity-check it's in the lower-left quadrant
    cx, cy = x + w/2, y + h/2
    print(f'CX={{cx:.0f}} CY={{cy:.0f}}')
    print(f'IN_LOWER_LEFT={{cx < 320 and cy > 240}}')
else:
    print(f'BBOX_MISSING={{bbox}}')
"""
        r = _run(script, timeout=120)
        # Model may refuse to ground (returns found=false) — skip is appropriate
        # because the test image is synthetic and qwen3.6 may treat it as abstract art.
        assert "FOUND=" in r.stdout, f"{r.stderr[-400:]}"
        if "FOUND=False" in r.stdout:
            pytest.skip("Model declined to ground on synthetic image")
        # Found → spatial correctness MUST hold. This previously used xfail which
        # silently masked regressions; we now assert so the next qwen3.x upgrade
        # will flag any grounding regression loudly.
        assert "IN_LOWER_LEFT=True" in r.stdout, (
            "Model returned bbox but not in the expected lower-left quadrant — "
            f"RefCOCO grounding regression on the test scene.\nstdout={r.stdout[-500:]}"
        )

    def test_count_items_returns_integer(self):
        b64 = _make_scene_jpeg_b64()
        script = f"""
import robot_tools as rt, robot_tools as m
m._get_frame_b64 = lambda: {b64!r}
r = rt.TOOLS['count_items'][1](description='circle')
print(f'COUNT={{r.get(\"count\")}}')
print(f'TYPE_OK={{isinstance(r.get(\"count\"), int)}}')
"""
        r = _run(script, timeout=120)
        assert "TYPE_OK=True" in r.stdout, f"{r.stderr[-400:]}"


# --- Parsing robustness -----------------------------------------------------

class TestParseBboxFromVLOutput:
    """The model may return bbox in varying formats. Our parser should tolerate."""
    def test_parse_plain_json_object(self):
        from robot_tools import _parse_bbox_response as p
        r = p('{"found": true, "bbox": [100, 200, 50, 60], "relative_position": "center"}')
        assert r["found"] is True
        assert r["bbox"] == [100, 200, 50, 60]

    def test_parse_with_markdown_fence(self):
        from robot_tools import _parse_bbox_response as p
        raw = '```json\n{"found": true, "bbox": [10, 20, 30, 40]}\n```'
        r = p(raw)
        assert r["found"] is True
        assert r["bbox"] == [10, 20, 30, 40]

    def test_parse_not_found(self):
        from robot_tools import _parse_bbox_response as p
        r = p('{"found": false}')
        assert r["found"] is False

    def test_parse_garbage_returns_not_found(self):
        from robot_tools import _parse_bbox_response as p
        r = p('sure here is the answer')
        assert r["found"] is False

    def test_parse_handles_trailing_comma(self):
        # Some models emit invalid trailing commas
        from robot_tools import _parse_bbox_response as p
        raw = '{"found": true, "bbox": [1, 2, 3, 4],}'
        r = p(raw)
        # lenient: either found with bbox or found=false after failed parse — don't crash
        assert isinstance(r, dict) and "found" in r


class TestParseCountFromVLOutput:
    def test_parse_plain_json(self):
        from robot_tools import _parse_count_response as p
        assert p('{"count": 3, "note": "three circles visible"}')["count"] == 3

    def test_parse_bare_number_fallback(self):
        from robot_tools import _parse_count_response as p
        # if model just says "3"
        r = p("3")
        assert r["count"] == 3

    def test_parse_natural_language(self):
        from robot_tools import _parse_count_response as p
        # "I see 3 circles"
        r = p("I see 3 circles in the image.")
        assert r["count"] == 3
