"""S5 — Tool registry for robot_brain.

Each tool is (spec, handler):
  - spec: OpenAI-compatible function schema (goes to Ollama in `tools` field)
  - handler: Python callable, returns JSON-serializable dict, never raises

Registry exposed as TOOLS = {name: (spec, handler)}.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import math as _math
import os as _os
import re as _re
import threading as _threading
import urllib.parse as _urlparse
import urllib.request as _urlreq
from typing import Any, Callable

REACHY_HOST = _os.getenv("REACHY_HOST", "100.85.191.3")
DAEMON_BASE = f"http://{REACHY_HOST}:8000"

# Vision model defaults — resolved per-call via _vision_endpoint() below so env
# changes at runtime take effect. Module-level constants kept only as the
# documented-default source; tests may monkey-patch them and will still win.
VISION_OLLAMA_URL = _os.getenv("VISION_URL", _os.getenv("OLLAMA_HOST", "http://localhost:11434"))
VISION_MODEL      = _os.getenv("VISION_MODEL", "qwen3.6:35b-a3b")

def _vision_endpoint() -> tuple[str, str]:
    """Return (base_url, model) honouring runtime env + any test monkey-patch.

    Resolution order for URL: VISION_URL > OLLAMA_HOST > module global > default.
    VISION_URL is preferred for the Moondream2 OpenAI-compat backend on
    vllm0528:8002; OLLAMA_HOST remains for back-compat with Ollama deployments.
    """
    g = globals()
    url = (
        _os.getenv("VISION_URL")
        or _os.getenv("OLLAMA_HOST")
        or g.get("VISION_OLLAMA_URL", "http://localhost:11434")
    )
    model = _os.getenv("VISION_MODEL", g.get("VISION_MODEL", "qwen3.6:35b-a3b"))
    return url, model


# ─────────────────────────── frame access (dep-injectable) ───────────────────
# robot_brain.py sets `_latest_frame` as a cv2 BGR ndarray. We avoid importing
# robot_brain here (heavy); instead each tool calls `_get_frame_b64()` which
# the brain can monkey-patch at startup. Default: try to import cv2 + brain lazily.
_STALE_FRAME_MAX_AGE_S = 2.0  # caller gets None → canned "no view" fallback


def _get_frame_b64() -> str | None:
    """Return current camera frame as base64 JPEG, or None if unavailable.

    Also returns None if the latest frame is older than ``_STALE_FRAME_MAX_AGE_S``
    (default 2.0 s). Mirrors the vision_worker recency guard in robot_brain
    (1.0 s there) — query_vision gets a slightly longer grace window since the
    LLM took some time to decide to call the tool, but a 2-second-old frame is
    the cap before we treat the camera as effectively offline.
    """
    try:
        import cv2
        import time as _time
        try:
            import robot_brain as _rb
        except ImportError:
            return None
        frame = getattr(_rb, "_latest_frame", None)
        if frame is None:
            return None
        last_t = getattr(_rb, "_latest_frame_t", 0.0) or 0.0
        # last_t == 0 means brain hasn't started camera worker yet → treat as stale
        if last_t <= 0 or (_time.time() - last_t) > _STALE_FRAME_MAX_AGE_S:
            return None
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        import base64
        return base64.b64encode(jpg.tobytes()).decode("ascii")
    except Exception:
        return None


def _ask_vision(prompt: str, b64_img: str, num_predict: int = 200,
                temperature: float = 0.2, timeout: float = 30) -> str:
    """Send an image + text prompt to the VL endpoint. Returns raw content string.

    Uses the OpenAI-compatible chat completions shape, which is served by the
    Moondream2 wrapper on vllm0528:8002 as well as upstream vLLM / OpenAI / any
    OpenAI-spec gateway. If the resolved URL already includes a path (e.g.
    ``http://host:8002/v1/chat/completions``) it is used verbatim, otherwise
    ``/v1/chat/completions`` is appended.

    The Moondream2 wrapper ignores ``temperature`` and treats ``max_tokens`` as
    ``max_new_tokens``; both fields are still sent for forward-compat with
    other backends.
    """
    url, model = _vision_endpoint()
    endpoint = url if "/chat/completions" in url else f"{url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": num_predict,
        "temperature": temperature,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ],
        }],
    }
    req = _urlreq.Request(
        endpoint,
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlreq.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if choices:
        return (choices[0].get("message", {}).get("content") or "").strip()
    # Tolerate Ollama-shape responses if a legacy backend is targeted.
    return (data.get("message", {}).get("content") or "").strip()


# ─────────────────────────────── parsers (unit-testable) ─────────────────────
_NUM_RE = _re.compile(r"\d+")

# Sanitize user-controlled strings before interpolating into VL prompts.
# Strip characters that let an attacker close the quote and inject fake JSON
# or markdown-fence out of a quoted region.
_SANITIZE_RE = _re.compile(r"[{}\[\]`<>]")
def _sanitize_description(desc: str, limit: int = 120) -> str:
    if not isinstance(desc, str):
        return ""
    cleaned = _SANITIZE_RE.sub(" ", desc).strip()
    # Also collapse runs of whitespace and cap length
    cleaned = _re.sub(r"\s+", " ", cleaned)[:limit]
    return cleaned

def _parse_bbox_response(raw: str) -> dict:
    """Parse VL response expected to look like:
        {"found": true, "bbox": [x, y, w, h], "relative_position": "..."}
       Tolerant to markdown fences, trailing commas, surrounding prose.
       Returns a dict with at least {"found": bool}. Never raises.
    """
    if not raw:
        return {"found": False}
    # Strip code fences
    txt = _re.sub(r"```(?:json)?|```", "", raw, flags=_re.IGNORECASE).strip()
    # Find first { and last }
    s, e = txt.find("{"), txt.rfind("}") + 1
    if s < 0 or e <= s:
        return {"found": False}
    body = txt[s:e]
    # Best-effort strip trailing commas before } or ]
    body = _re.sub(r",\s*(\}|\])", r"\1", body)
    try:
        obj = _json.loads(body)
    except Exception:
        return {"found": False}
    if not isinstance(obj, dict):
        return {"found": False}
    # Normalize
    found = bool(obj.get("found"))
    out = {"found": found}
    bbox = obj.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        out["bbox"] = [int(v) for v in bbox]
    rp = obj.get("relative_position")
    if isinstance(rp, str):
        out["relative_position"] = rp
    return out


def _parse_count_response(raw: str, query: str = "") -> dict:
    """Extract a count integer from VL output.

    Priority:
      1. Well-formed JSON with a numeric `count` — use that directly.
      2. If `query` is provided, look for `<N> ... <query>` or `<query> ... <N>`
         patterns first (e.g. "3 cups" / "cups: 3") — avoids the "0 cups but 3
         plates" failure mode where the first integer belongs to a different noun.
      3. First integer anywhere in the text.
      4. Give up with count=0.
    """
    if not raw:
        return {"count": 0, "note": "(empty)"}
    txt = _re.sub(r"```(?:json)?|```", "", raw, flags=_re.IGNORECASE).strip()
    # 1. JSON path
    s, e = txt.find("{"), txt.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            obj = _json.loads(txt[s:e])
            if isinstance(obj, dict) and isinstance(obj.get("count"), (int, float)):
                return {"count": int(obj["count"]), "note": str(obj.get("note", ""))}
        except Exception:
            pass
    # 2. Query-aware — pick the integer whose position is NEAREST the query term.
    # Distance-based beats regex matching because greedy regex can jump across
    # the wrong noun (e.g. "0 cups but 3 plates" with query=plates must NOT
    # greedily bridge "0 cups but 3 plates" and return 0).
    if query:
        q = _re.escape(query.strip())
        q_m = _re.search(rf"\b{q}\b", txt, _re.IGNORECASE)
        if q_m:
            q_center = (q_m.start() + q_m.end()) // 2
            nums = list(_re.finditer(r"\b(\d+)\b", txt))
            if nums:
                best = min(nums, key=lambda m: abs(((m.start() + m.end()) // 2) - q_center))
                return {"count": int(best.group(1)), "note": txt[:120]}
    # 3. First integer
    m = _NUM_RE.search(txt)
    if m:
        return {"count": int(m.group(0)), "note": txt[:120]}
    return {"count": 0, "note": txt[:120]}


# ──────────────────────────── individual tool impls ─────────────────────────

def _tool_get_current_time(**_kwargs) -> dict:
    """Return ISO-8601 local time, plus a human-friendly string."""
    try:
        now = _dt.datetime.now().astimezone()
        return {
            "now": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "iso": now.isoformat(timespec="seconds"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _post_daemon(path: str, body: dict | None = None, timeout: float = 5.0) -> dict:
    try:
        req = _urlreq.Request(
            f"{DAEMON_BASE}{path}",
            data=_json.dumps(body or {}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        try:
            return {"ok": True, "response": _json.loads(raw.decode("utf-8"))}
        except Exception:
            return {"ok": True, "response": raw.decode("utf-8", errors="replace")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _tool_stop_motion(**_kwargs) -> dict:
    """Halt all current motion; disable motors (head will droop softly)."""
    return _post_daemon("/api/motors/set_mode/disabled")


def _tool_move_head(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0,
                    duration: float | None = None,
                    **_kwargs) -> dict:
    """Move head to (pitch, yaw, roll) in degrees. Safe range: ±25° per axis.

    Always uses smooth minjerk interpolation with duration clamped to at least
    MOTOR_MIN_DURATION_S seconds. Instant motion is forbidden — the pre-recorded
    look_around path was deprecated because its speed startles users and stresses
    the motors (memory: reference_reachy_mini_wireless_motor.md).

    Values outside ±25° are clamped silently (not an error — LLM may overshoot).
    """
    try:
        p = float(pitch); y = float(yaw); r = float(roll)
    except (TypeError, ValueError) as e:
        return {"error": f"invalid args: {e}"}
    clipped = False
    def _clip(v, lo=-25.0, hi=25.0):
        nonlocal clipped
        if v < lo: clipped = True; return lo
        if v > hi: clipped = True; return hi
        return v
    p, y, r = _clip(p), _clip(y), _clip(r)
    # SAFETY: enforce minimum duration. Anything below MOTOR_MIN_DURATION_S
    # (env, default 1.8s) is raised up to it. LLM may pass duration=0 or
    # omit it — clamp always ends up safe.
    MIN_DUR = float(_os.getenv("MOTOR_MIN_DURATION_S", "1.8"))
    try:
        d = max(MIN_DUR, float(duration) if duration is not None else MIN_DUR)
    except (TypeError, ValueError):
        d = MIN_DUR
    # UNIT FIX (2026-06-01): daemon /api/move/goto XYZRPYPose expects
    # roll/pitch/yaw in RADIANS — OpenAPI schema says "angles in radians", and
    # this was verified live (yaw=0.35 rad turns head ~20°; yaw=15 raw → +134°
    # runaway, out of the Stewart platform's working range → head wouldn't move
    # = the user's "look right does nothing"). The whole LLM-facing API + the
    # ±25° clamp above are in DEGREES, so convert ONLY here at the REST boundary.
    # note_head_command below stays in degrees (face-tracker baseline is degrees).
    body = {
        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                      "roll": _math.radians(r),
                      "pitch": _math.radians(p),
                      "yaw": _math.radians(y)},
        "antennas": None,
        "body_yaw": None,
        "duration": d,
        "interpolation": "minjerk",
    }
    # Always log the attempt (audit even on failure).
    try:
        from motor_log import log_motor
        log_motor("tool_move_head", caller="llm_tool_call",
                  pitch=p, yaw=y, roll=r, duration=d, clipped=clipped)
    except Exception:
        pass
    result = _post_daemon("/api/move/goto", body)
    # Only sync face-tracker baseline AFTER daemon accepts. If daemon rejects
    # (network / busy / motor torque OFF), motor never moved — updating the
    # baseline would make face tracker race toward a phantom pose. Also pass
    # body_yaw_rad=None: this handler doesn't touch body yaw, so the baseline
    # for body must stay at whatever was last actually commanded.
    if isinstance(result, dict) and result.get("ok"):
        try:
            import robot_brain as _rb
            _rb.note_head_command(pitch_deg=p, yaw_deg=y, body_yaw_rad=None,
                                  source="tool_move_head")
            # 抑制 face tracker：手勢執行(d) + hold 期間不讓追蹤把頭轉回置中。
            # 這是「看左右(yaw)沒反應」的永久修——yaw 軸人臉留在畫面內、tracker
            # 會在 ~50ms 內轉回置中抵銷 move_head。pitch/roll 較少被搶所以偶爾成功。
            _rb.pause_tracking(d + getattr(_rb, "GESTURE_TRACK_HOLD_S", 2.0))
        except Exception:
            pass
        result["clipped"] = clipped
        result["duration_s"] = d
    return result


_resync_timer_lock = _threading.Lock()
_resync_timer: _threading.Timer | None = None
# H6 fix (2026-06-01): monotonic generation counter for last-write-wins
# semantics. Timer.cancel() is best-effort — if the timer already fired and
# _do_sync is mid-execution (e.g. blocked on the daemon HTTP call) when a
# newer clip schedules a fresh timer, the old _do_sync would still write
# a stale baseline AFTER the cancel + after the new timer was created.
# Each scheduling bumps the counter and snapshots the new value; _do_sync
# refuses to write the baseline if its snapshot is no longer the latest.
_resync_gen: int = 0


def _schedule_face_baseline_resync(delay_s: float = 5.0, source: str = "post_clip") -> None:
    """Schedule a debounced one-shot baseline resync after a recorded-move
    clip is expected to finish. Queries the daemon for the actual head pose
    and writes it into robot_brain's face-tracker baseline. Without this,
    the baseline stays at its pre-clip value while the head physically
    elsewhere — the next face-tracker frame races toward the stale baseline
    causing a 20-30° instant snap (the symptom the delta-clamp was meant
    to prevent).

    Debounce: if a previous resync timer is still pending, cancel it. A
    second clip fired before the first resync completes supersedes the
    first — only the LAST clip's expected end time matters. Timer.cancel()
    only stops timers that haven't fired yet; an already-running _do_sync
    is gated by the gen-counter check below so it aborts before writing.

    Daemon endpoint: GET /api/state/present_head_pose → {pitch, yaw, roll}.
    On failure (network / unknown format), falls back to neutral (0, 0) —
    most pollen-robotics clips return to neutral, so this is a reasonable
    last-resort default.
    """
    # 2026-06-01: 暫停 face tracker 直到 clip 大致播完 + hold。否則 tracker 會在
    # clip 還在播時每幀送 set_target 搶馬達、把舞蹈/情緒動作蓋掉（同「看左右被
    # 蓋掉」根因）。這裡一處涵蓋全部 emotion/dance 播放路徑（都會呼叫本函式）。
    try:
        import robot_brain as _rb
        _rb.pause_tracking(delay_s + getattr(_rb, "GESTURE_TRACK_HOLD_S", 2.0))
    except Exception:
        pass
    global _resync_timer, _resync_gen
    with _resync_timer_lock:
        # Cancel any pending resync — last clip wins, no overlapping writes.
        if _resync_timer is not None:
            try:
                _resync_timer.cancel()
            except Exception:
                pass
        _resync_gen += 1
        my_gen = _resync_gen

        def _do_sync(my_gen: int = my_gen) -> None:
            pitch, yaw = 0.0, 0.0
            try:
                req = _urlreq.Request(f"{DAEMON_BASE}/api/state/present_head_pose")
                with _urlreq.urlopen(req, timeout=3) as resp:
                    pose = _json.loads(resp.read().decode("utf-8"))
                if isinstance(pose, dict):
                    pitch = float(pose.get("pitch", 0.0))
                    yaw   = float(pose.get("yaw", 0.0))
            except Exception:
                pass  # fall through to neutral baseline
            # Re-check generation under the lock right before writing. If a
            # newer scheduling has happened (user fired another clip while we
            # were blocked on the daemon HTTP call), abort this stale write.
            with _resync_timer_lock:
                if my_gen != _resync_gen:
                    return
            try:
                import robot_brain as _rb
                _rb.note_head_command(pitch_deg=pitch, yaw_deg=yaw,
                                      body_yaw_rad=None, source=source)
            except Exception:
                pass

        _resync_timer = _threading.Timer(max(0.5, float(delay_s)), _do_sync)
        _resync_timer.daemon = True
        _resync_timer.start()

# Catalog of emotion + dance clips lives in src/motion_catalog.py (Wave4-P6 #76).
# It refreshes from HuggingFace once per 24h with disk + in-process cache and
# falls back to data/catalog_fallback.json if HF is unreachable. The exported
# MOTION_CATALOG dict maps every triggerable name (clips + aliases + generic
# "dance" synonyms) to a MotionSpec with the daemon play path.
#
# We keep the original _DANCES_LIB / _EMOTIONS_LIB / _EMOTION_ALIAS / helper
# names as thin shims so motion-queue work (#73) and any other call site that
# imports them keeps working unchanged.
try:
    from motion_catalog import (  # type: ignore
        DANCES_DATASET as _DANCES_LIB,
        EMOTIONS_DATASET as _EMOTIONS_LIB,
        MOTION_CATALOG,
    )
except ImportError:  # pragma: no cover — package-style fallback
    from .motion_catalog import (
        DANCES_DATASET as _DANCES_LIB,
        EMOTIONS_DATASET as _EMOTIONS_LIB,
        MOTION_CATALOG,
    )

# Legacy fallback set kept for any external import; catalog handles the real
# fallback now.
_DANCE_NAMES_FALLBACK = {
    "yeah_nod", "simple_nod", "side_to_side_sway", "groovy_sway_and_roll",
    "jackson_square", "dizzy_spin", "pendulum_swing", "chicken_peck",
}


def _fetch_dataset_moves(dataset_full_name: str) -> set[str]:
    """Legacy shim: return the set of clip names for ``dataset_full_name``
    from MOTION_CATALOG. Kept so any out-of-tree caller still works."""
    return {
        spec.clip for spec in MOTION_CATALOG.values()
        if spec.dataset == dataset_full_name and spec.kind in {"emotion", "dance"}
    }


def _get_dance_names() -> set[str]:
    discovered = _fetch_dataset_moves(_DANCES_LIB)
    return discovered if discovered else _DANCE_NAMES_FALLBACK


def _get_emotion_clip_names() -> set[str]:
    return _fetch_dataset_moves(_EMOTIONS_LIB)


# Legacy export: LLM-friendly English names → emotions-library clip names.
# Backed by the catalog so it auto-stays in sync; kept as a module-level dict
# for any caller that imports it directly.
_EMOTION_ALIAS = {
    spec.name: spec.clip
    for spec in MOTION_CATALOG.values()
    if spec.kind == "alias"
}


def _tool_play_emotion(name: str = "", **_kwargs) -> dict:
    """Play a pre-recorded animation (emotion OR dance) from official HF libraries.

    Routing (matches implementation order):
      1. name in dances library (19 names: yeah_nod / chicken_peck / chin_lead / ...)
         → play directly from pollen-robotics/reachy-mini-dances-library
      2. name == "dance" / "dancing" / "boogie" / "groove" / "jig" → default dance "yeah_nod"
      3. name matches an emotions-library clip exactly (e.g. dance3, cheerful2,
         enthusiastic2, proud3, yes_sad1, oops2, understanding2)
         → play directly from pollen-robotics/reachy-mini-emotions-library
      4. name in alias map (happy / cheerful / sad / curious / proud / loving ...)
         → translate to clip and play from emotions library
      5. otherwise → error with hint of valid names

    NOTE: clip playback speed is controlled by the recorded JSON timeline on the
    daemon, NOT by this handler. The face tracker delta clamp in robot_brain.py
    still protects against the racing-back-to-face side-effect after the clip
    finishes — that was the only bit we could fix on the LLM side.
    """
    n = (name or "").strip().lower()
    if not n:
        return {"error": "empty emotion/dance name"}

    dances   = _get_dance_names()
    emotions = _get_emotion_clip_names()

    # All four playback branches below schedule a baseline resync ~5 s after
    # firing so the face tracker doesn't race back to its stale pre-clip
    # baseline. See _schedule_face_baseline_resync for rationale.

    # 1. Direct dance-library hit (dynamic — covers any name pollen publishes)
    if n in dances:
        ds = _urlparse.quote(_DANCES_LIB, safe="")
        result = _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{n}")
        if isinstance(result, dict) and result.get("ok"):
            _schedule_face_baseline_resync(source=f"after_dance:{n}")
        return result

    # 2. Generic "dance" intent → prefer "yeah_nod" (short + friendly default);
    #    fall back to alphabetically-first available dance if yeah_nod is gone.
    if n in {"dance", "dancing", "boogie", "groove", "jig"}:
        pick = "yeah_nod" if "yeah_nod" in dances else (sorted(dances)[0] if dances else "yeah_nod")
        ds = _urlparse.quote(_DANCES_LIB, safe="")
        result = _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{pick}")
        if isinstance(result, dict) and result.get("ok"):
            _schedule_face_baseline_resync(source=f"after_dance:{pick}")
        return result

    # 3. Direct emotion-library hit (dynamic — covers ALL clips, e.g. dance3,
    #    enthusiastic2, proud3, yes_sad1, oops2, understanding2 — beyond aliases)
    if n in emotions:
        ds = _urlparse.quote(_EMOTIONS_LIB, safe="")
        result = _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{n}")
        if isinstance(result, dict) and result.get("ok"):
            _schedule_face_baseline_resync(source=f"after_emotion:{n}")
        return result

    # 4. English-friendly alias (LLM emits "happy" → cheerful1, etc.)
    if n in _EMOTION_ALIAS:
        clip = _EMOTION_ALIAS[n]
        ds = _urlparse.quote(_EMOTIONS_LIB, safe="")
        result = _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{clip}")
        if isinstance(result, dict) and result.get("ok"):
            _schedule_face_baseline_resync(source=f"after_emotion:{clip}")
        return result

    # 5. Unknown — hint at available categories without dumping 100 names
    hint_dances = sorted(list(dances))[:6] if dances else []
    return {"error": f"unknown name {name!r}",
            "hint_dances": hint_dances,
            "hint_aliases": ["happy", "sad", "curious", "thoughtful", "greet",
                             "nod", "shake", "surprised", "scared", "proud",
                             "loving", "tired", "confused", "calm"]}


def _tool_see_what(query: str = "", **_kwargs) -> dict:
    """Caption the robot's current camera frame. Optional query narrows focus."""
    b64 = _get_frame_b64()
    if b64 is None:
        return {"error": "no frame available — camera worker may not be running"}
    if query:
        prompt = (f"You are looking through the robot's camera. "
                  f"Answer this focused question in 1-2 sentences based on the image: {query}")
    else:
        prompt = ("You are looking through the robot's camera. "
                  "Describe what you see in 1-2 short sentences. Focus on the most "
                  "salient object or person; mention colors, positions, and actions.")
    try:
        text = _ask_vision(prompt, b64, num_predict=180, temperature=0.25)
    except Exception as e:
        return {"error": f"vision call failed: {type(e).__name__}: {e}"}
    if not text:
        return {"error": "vision model returned empty"}
    return {"description": text}


def _tool_find_in_view(description: str = "", **_kwargs) -> dict:
    """Locate `description` in the current frame; return bbox + relative position."""
    description = _sanitize_description(description)
    if not description:
        return {"error": "description argument is required"}
    b64 = _get_frame_b64()
    if b64 is None:
        return {"error": "no frame available"}
    prompt = (
        f"Locate '{description}' in this image.\n"
        f"Reply ONLY with a JSON object — no markdown fence, no prose:\n"
        f"  {{\"found\": true, \"bbox\": [x, y, w, h], \"relative_position\": \"center|left|right|upper|lower|upper-left|upper-right|lower-left|lower-right\"}}\n"
        f"If not visible, reply {{\"found\": false}}.\n"
        f"Use top-left origin; x,y is bbox top-left corner; w,h are width and height in pixels."
    )
    try:
        raw = _ask_vision(prompt, b64, num_predict=120, temperature=0.1)
    except Exception as e:
        return {"error": f"vision call failed: {type(e).__name__}: {e}"}
    parsed = _parse_bbox_response(raw)
    parsed["query"] = description
    return parsed


def _tool_count_items(description: str = "", **_kwargs) -> dict:
    """Count how many instances of `description` are visible in the current frame."""
    description = _sanitize_description(description)
    if not description:
        return {"error": "description argument is required"}
    b64 = _get_frame_b64()
    if b64 is None:
        return {"error": "no frame available"}
    prompt = (
        f"Count how many '{description}' are visible in this image.\n"
        f"Reply ONLY with a JSON object — no markdown fence:\n"
        f"  {{\"count\": N, \"note\": \"brief observation\"}}\n"
        f"If you cannot see the scene clearly, reply {{\"count\": 0, \"note\": \"unclear\"}}."
    )
    try:
        raw = _ask_vision(prompt, b64, num_predict=120, temperature=0.1)
    except Exception as e:
        return {"error": f"vision call failed: {type(e).__name__}: {e}"}
    out = _parse_count_response(raw, query=description)
    out["query"] = description
    return out


def _tool_query_vision(question: str = "", **_kwargs) -> dict:
    """Ask the VLM a specific, on-demand question about the current camera frame.

    Distinct from `see_what` (which is optional-query + 1-2 sentence caption):
    `query_vision` REQUIRES a `question` and returns a focused, short answer
    suitable for direct synthesis by the dialog LLM. Use when the periodic
    LATEST_SCENE_DESC (every VISION_INTERVAL s, generic 4-part prompt) does
    not cover the user's question — e.g. "is the cup on the table?",
    "what is the user holding in their right hand?".

    Sync from LLM's perspective; 20 s timeout cap so a hung VLM cannot freeze
    the brain. Stateless (no caching) — keeps Pi RAM footprint flat.

    Metrics + structured log emitted by the brain dispatcher (which wraps
    `execute_tool`) so this helper stays Pi-cheap and import-light.
    """
    question = (question or "").strip()
    if not question:
        return {"error": "question argument is required"}
    b64 = _get_frame_b64()
    if b64 is None:
        # Canned answer so the LLM can synthesize a graceful fallback instead
        # of a JSON-shaped error leaking to the user.
        return {"answer": "I can't see anything right now — the camera isn't ready."}
    # Sanitize to keep injected prose out of the VL prompt envelope.
    safe_q = _sanitize_description(question, limit=240)
    prompt = (
        "You are looking through the robot's camera. Answer the following "
        "question in 1-2 short, concrete sentences based ONLY on what is "
        f"visible in this image:\n{safe_q}\n"
        "If the answer is not visible, say so plainly."
    )
    try:
        raw = _ask_vision(prompt, b64, num_predict=120, temperature=0.2, timeout=20)
    except Exception as e:
        return {"error": f"vision call failed: {type(e).__name__}: {e}"}
    if not raw:
        return {"error": "vision model returned empty"}
    return {"answer": raw.strip(), "question": question}


def _tool_recall_memory(query: str = "", max_results: int = 3, **_kwargs) -> dict:
    """Search conversation_log.jsonl for turns whose user-or-robot text contains
    `query` (case-insensitive substring). Returns up to `max_results` most recent."""
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "conversation_log.jsonl")
    q = (query or "").strip().lower()
    if not q:
        return {"error": "empty query"}
    hits = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                blob = (rec.get("user", "") + " " + rec.get("robot", "")).lower()
                if q in blob:
                    hits.append({
                        "ts": rec.get("ts"),
                        "user": rec.get("user", "")[:200],
                        "robot": rec.get("robot", "")[:200],
                    })
    except FileNotFoundError:
        return {"error": "no conversation_log yet"}
    hits = hits[-max_results:] if max_results > 0 else hits
    return {"matches": hits, "query": q, "count": len(hits)}


# ─────────────────────────────── specs (OpenAI fmt) ──────────────────────────

def _spec(name: str, desc: str, properties: dict, required: list[str] | None = None) -> dict:
    params = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params},
    }


TOOLS: dict[str, tuple[dict, Callable[..., dict]]] = {
    "get_current_time": (
        _spec("get_current_time",
              "Return the current wall-clock time. Use when the user asks what time it is.",
              {}),
        _tool_get_current_time,
    ),
    "stop_motion": (
        _spec("stop_motion",
              "Disable the robot's motors immediately. Use when the user says stop, be still, or the robot should rest.",
              {}),
        _tool_stop_motion,
    ),
    "move_head": (
        _spec("move_head",
              "Move the robot head smoothly to a pose in degrees. All axes optional, default 0. Safe range ±25°.",
              {
                  "pitch":    {"type": "number", "description": "Head tilt up/down in degrees (+down)."},
                  "yaw":      {"type": "number", "description": "Head turn left/right in degrees (+right)."},
                  "roll":     {"type": "number", "description": "Head tilt sideways in degrees."},
                  "duration": {"type": "number", "description": "Optional seconds for the move. Floor of ~1.8s is enforced for safety; pass a larger value for a slower, more expressive motion (e.g. 3.0 for a deliberate look)."},
              }),
        _tool_move_head,
    ),
    "play_emotion": (
        _spec("play_emotion",
              "Play a pre-recorded animation: emotion OR dance. "
              "Pass 'dance' for a generic dance, or any specific clip name from "
              "the official pollen-robotics reachy-mini-{dances,emotions}-library "
              "datasets (discovered dynamically from the daemon at runtime; "
              "common aliases like happy/sad/curious/proud/loving/scared/tired/"
              "confused/calm/welcome map to canonical emotion clips). "
              "Examples — dances: yeah_nod, jackson_square, dizzy_spin, "
              "groovy_sway_and_roll. Emotions: happy, sad, curious, surprised.",
              {"name": {"type": "string", "description": "Animation name. Use 'dance' for a generic dance, or a specific clip name (dance or emotion). Unknown names return an error with hints."}},
              required=["name"]),
        _tool_play_emotion,
    ),
    "recall_memory": (
        _spec("recall_memory",
              "Search past conversation history for a keyword. Use when user asks 'do you remember…' or references prior context that isn't in the current window.",
              {
                  "query":       {"type": "string",  "description": "Substring to search for (case-insensitive)."},
                  "max_results": {"type": "integer", "description": "Max matches to return, default 3."},
              },
              required=["query"]),
        _tool_recall_memory,
    ),
    # ── vision tools (Qwen3.6 native VL: MMMU 81.7, RefCOCO 92) ──────────────
    "see_what": (
        _spec("see_what",
              "Look through the robot's camera and give a 1-2 sentence ambient description "
              "of what is currently visible (people, surroundings, mood). Use this for "
              "general/open-ended 'what do you see' or 'what's in front of you' updates. "
              "For a SPECIFIC factual question (counting, identifying held objects, reading "
              "text, precise spatial checks) use query_vision instead — it forces a fresh "
              "focused inference. Optional query narrows the caption topic only.",
              {"query": {"type": "string", "description": "Optional topic to bias the ambient caption (e.g. 'focus on people'). For specific factual questions, use query_vision instead."}}),
        _tool_see_what,
    ),
    "find_in_view": (
        _spec("find_in_view",
              "Locate a specific object or person in the current camera view. "
              "Returns pixel bounding box and relative position (left/right/upper/lower). "
              "Use when the user references 'the red mug', 'the person on the left', etc.",
              {"description": {"type": "string", "description": "What to look for, e.g. 'the red mug' or 'a person wearing glasses'."}},
              required=["description"]),
        _tool_find_in_view,
    ),
    "count_items": (
        _spec("count_items",
              "Count the number of instances of a class of object in the current camera view.",
              {"description": {"type": "string", "description": "What class of item to count, e.g. 'people', 'books', 'pens'."}},
              required=["description"]),
        _tool_count_items,
    ),
    # Wave6-P4 (2026-05-29): on-demand precise visual query. Distinct from
    # see_what (optional query, 1-2 sentence caption) — query_vision REQUIRES
    # a question and is the preferred tool when the user asks something
    # specific that the periodic LATEST_SCENE_DESC may not cover.
    "query_vision": (
        _spec("query_vision",
              "Ask the vision model a SPECIFIC factual question requiring a FRESH on-demand "
              "inference about the current view — counting items, identifying objects "
              "currently held in someone's hand, reading visible text/labels, verifying a "
              "precise spatial relation (e.g. 'how many fingers am I holding up', 'what is "
              "the user holding in their right hand', 'what does the label on the bottle "
              "say', 'is the red cup to the left of the laptop'). Use this when accuracy "
              "matters NOW and the periodic cached scene description may be stale or too "
              "generic. For an ambient/general 'what do you see' update, prefer see_what "
              "instead. Returns {answer: string}.",
              {"question": {"type": "string", "description": "The specific factual question to ask the vision model about the current view."}},
              required=["question"]),
        _tool_query_vision,
    ),
}


# ──────────────────────────── runtime helpers ────────────────────────────────

def get_tool_specs() -> list[dict]:
    """Return the list of tool specs to send to Ollama."""
    return [spec for spec, _ in TOOLS.values()]


def parse_tool_calls(message: dict) -> list[dict]:
    """Normalize Ollama's `tool_calls` field into [{name, arguments}, …].
    Handles both dict-arguments and JSON-string-arguments (llama.cpp variants)."""
    out = []
    for tc in (message or {}).get("tool_calls", []) or []:
        fn = tc.get("function", {}) or {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if name:
            out.append({"name": name, "arguments": args})
    return out


def execute_tool(name: str, arguments: dict) -> dict:
    """Look up + run a tool. Never raises; unknown tool / bad args → {error}."""
    entry = TOOLS.get(name)
    if entry is None:
        return {"error": f"unknown tool {name!r}"}
    _, handler = entry
    try:
        return handler(**(arguments or {}))
    except TypeError as e:
        # wrong kwargs — return as error, don't crash
        return {"error": f"bad arguments: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
