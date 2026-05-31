"""S3 TDD: qwen3.6:35b-a3b native vision vs qwen2.5vl:7b baseline.

Pass criteria:
  - Both models return non-empty captions on the same JPEG.
  - qwen3.6 latency per call ≤ 1.5× qwen2.5vl baseline (it's 5× bigger total).
  - qwen3.6 response describes at least 2 of 3 ground-truth elements
    (red_square, blue_circle, text "HELLO") — proves visual capability not fake.

Red phase: no implementation to pass yet — this is a verification test.
If qwen3.6 fails → keep qwen2.5vl, mark S3 skipped.
If qwen3.6 passes → proceed to swap worker model.
"""
import base64
import io
import json
import time
import urllib.request
import pytest
from tests._helpers import ollama_running, pillow_available

pytestmark = [
    pytest.mark.skipif(not ollama_running(), reason="Ollama not reachable"),
    pytest.mark.skipif(not pillow_available(), reason="Pillow not installed — install .[dev]"),
]

OLLAMA = "http://localhost:11434"

# ---- Fixture: generate a test image with known content ----
@pytest.fixture(scope="module")
def test_image_b64():
    """繪製有 ground truth 的測試圖：red square / blue circle / HELLO 文字"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (640, 480), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    # red square bottom-left
    d.rectangle([50, 300, 200, 450], fill=(220, 30, 30))
    # blue circle top-right
    d.ellipse([420, 50, 590, 220], fill=(30, 80, 220))
    # "HELLO" text center
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except Exception:
        font = ImageFont.load_default()
    d.text((230, 210), "HELLO", fill=(20, 20, 20), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), buf.getvalue()

def _ask_vision(model, b64_img, prompt="Describe what you see in one short sentence.", timeout=120):
    payload = {
        "model": model,
        "stream": False,
        "think": False,                       # ← 關鍵：qwen3.6 capability 含 thinking，預設 on 會吃 num_predict
        "keep_alive": "2m",
        "options": {"num_ctx": 2048, "num_predict": 200, "temperature": 0.2},
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [b64_img],
        }],
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    wall_ms = (time.perf_counter() - t0) * 1000
    content = (data.get("message", {}).get("content") or "").strip()
    return content, wall_ms

def _unload(model):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{OLLAMA}/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        ), timeout=10)
    except Exception:
        pass

# ---- Baseline qwen2.5vl:7b retired 2026-04-21 (test deleted 2026-06-01, Track D-1)
# ---- See git history for the original comparison that chose qwen3.6 as the vision worker.

# ---- Candidate: qwen3.6:35b-a3b ----
def test_qwen36_vision(test_image_b64):
    b64, _ = test_image_b64
    _unload("qwen3.6:35b-a3b")
    time.sleep(1)
    caption, wall = _ask_vision("qwen3.6:35b-a3b", b64)
    print(f"\n[qwen3.6:35b-a3b] {wall:.0f}ms → {caption}")
    assert caption, "qwen3.6 vision returned empty"

    # Ground truth recall: at least 2 of 3 elements mentioned
    cap_low = caption.lower()
    ground_hits = sum([
        any(k in cap_low for k in ["red", "square", "rectangle"]),
        any(k in cap_low for k in ["blue", "circle", "round", "ball", "dot"]),
        "hello" in cap_low,
    ])
    assert ground_hits >= 2, f"qwen3.6 only identified {ground_hits}/3 elements: {caption}"

# ---- test_latency_comparison deleted 2026-06-01 (Track D-1): qwen2.5vl:7b retired
# ---- 2026-04-21, no baseline to compare against. See git history if needed.
