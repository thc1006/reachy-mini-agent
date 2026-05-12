"""S5 — Tool registry for robot_brain.

Each tool is (spec, handler):
  - spec: OpenAI-compatible function schema (goes to Ollama in `tools` field)
  - handler: Python callable, returns JSON-serializable dict, never raises

Registry exposed as TOOLS = {name: (spec, handler)}.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import re as _re
import urllib.request as _urlreq
from typing import Any, Callable

REACHY_HOST = _os.getenv("REACHY_HOST", "100.85.191.3")
DAEMON_BASE = f"http://{REACHY_HOST}:8000"

# Vision model defaults — resolved per-call via _vision_endpoint() below so env
# changes at runtime take effect. Module-level constants kept only as the
# documented-default source; tests may monkey-patch them and will still win.
VISION_OLLAMA_URL = _os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL      = _os.getenv("VISION_MODEL", "qwen3.6:35b-a3b")

def _vision_endpoint() -> tuple[str, str]:
    """Return (base_url, model) honouring runtime env + any test monkey-patch."""
    g = globals()
    url   = _os.getenv("OLLAMA_HOST", g.get("VISION_OLLAMA_URL", "http://localhost:11434"))
    model = _os.getenv("VISION_MODEL", g.get("VISION_MODEL", "qwen3.6:35b-a3b"))
    return url, model


# ─────────────────────────── frame access (dep-injectable) ───────────────────
# robot_brain.py sets `_latest_frame` as a cv2 BGR ndarray. We avoid importing
# robot_brain here (heavy); instead each tool calls `_get_frame_b64()` which
# the brain can monkey-patch at startup. Default: try to import cv2 + brain lazily.
def _get_frame_b64() -> str | None:
    """Return current camera frame as base64 JPEG, or None if unavailable."""
    try:
        import cv2
        try:
            import robot_brain as _rb
        except ImportError:
            return None
        frame = getattr(_rb, "_latest_frame", None)
        if frame is None:
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
    """Send an image + text prompt to the Qwen3.6 VL endpoint. Returns raw content
    string (may contain markdown fence, etc.)."""
    url, model = _vision_endpoint()
    payload = {
        "model": model,
        "stream": False,
        "think": False,   # critical — qwen3.6 thinking eats num_predict otherwise
        "keep_alive": "30m",
        "options": {"num_ctx": 2048, "num_predict": num_predict, "temperature": temperature},
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [b64_img],
        }],
    }
    req = _urlreq.Request(
        f"{url}/api/chat",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlreq.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
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
    # SAFETY: enforce minimum duration. duration<1.5s is rejected up to 1.5s.
    # LLM may try to set duration=0 or omit it — we always end up safe.
    MIN_DUR = float(_os.getenv("MOTOR_MIN_DURATION_S", "1.8"))
    try:
        d = max(MIN_DUR, float(duration) if duration is not None else MIN_DUR)
    except (TypeError, ValueError):
        d = MIN_DUR
    body = {
        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": r, "pitch": p, "yaw": y},
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
        except Exception:
            pass
        result["clipped"] = clipped
        result["duration_s"] = d
    return result


import urllib.parse as _urlparse
import threading as _threading

_DANCES_LIB   = "pollen-robotics/reachy-mini-dances-library"
_EMOTIONS_LIB = "pollen-robotics/reachy-mini-emotions-library"

# Dynamic discovery — query daemon's list endpoint at first use, cache result.
# This way the catalog auto-syncs with whatever pollen-robotics publishes and
# whatever extra datasets the user installs on the daemon. No hardcoded names.
#
# Fallback set is a minimal known-good list in case the daemon is unreachable
# at module load time (then call retries on next invocation).
_DANCE_NAMES_FALLBACK = {
    "yeah_nod", "simple_nod", "side_to_side_sway", "groovy_sway_and_roll",
    "jackson_square", "dizzy_spin", "pendulum_swing", "chicken_peck",
}
_dataset_cache: dict[str, set[str]] = {}
_dataset_cache_lock = _threading.Lock()


def _fetch_dataset_moves(dataset_full_name: str) -> set[str]:
    """Return the set of move names available in ``dataset_full_name`` on the
    daemon. Cached after first successful call. Empty set on failure."""
    with _dataset_cache_lock:
        cached = _dataset_cache.get(dataset_full_name)
    if cached is not None:
        return cached
    encoded = _urlparse.quote(dataset_full_name, safe="")
    url = f"{DAEMON_BASE}/api/move/recorded-move-datasets/list/{encoded}"
    try:
        req = _urlreq.Request(url)
        with _urlreq.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        if isinstance(data, list):
            result = set(data)
            with _dataset_cache_lock:
                _dataset_cache[dataset_full_name] = result
            return result
    except Exception:
        pass
    return set()


def _get_dance_names() -> set[str]:
    discovered = _fetch_dataset_moves(_DANCES_LIB)
    return discovered if discovered else _DANCE_NAMES_FALLBACK


def _get_emotion_clip_names() -> set[str]:
    return _fetch_dataset_moves(_EMOTIONS_LIB)

# LLM-friendly English names → emotions-library clip names. 80+ clips
# available; this maps the common English emotion words an LLM would emit.
_EMOTION_ALIAS = {
    "happy": "cheerful1", "cheerful": "cheerful1", "joyful": "laughing1",
    "laugh": "laughing1", "laughing": "laughing1",
    "sad": "sad1", "lonely": "lonely1", "downcast": "downcast1",
    "curious": "curious1", "think": "thoughtful1", "thoughtful": "thoughtful1",
    "greet": "welcoming1", "welcome": "welcoming1", "hello": "welcoming1",
    "shake": "no1", "no": "no1",
    "nod": "yes1", "yes": "yes1",
    "surprised": "surprised1", "amazed": "amazed1",
    "scared": "scared1", "fear": "fear1", "anxious": "anxiety1",
    "proud": "proud1", "success": "success1", "victory": "success1",
    "tired": "tired1", "exhausted": "exhausted1", "sleep": "sleep1",
    "loving": "loving1", "love": "loving1", "grateful": "grateful1",
    "confused": "confused1", "uncertain": "uncertain1",
    "shy": "shy1", "embarrassed": "shy1",
    "frustrated": "frustrated1", "angry": "rage1", "rage": "rage1",
    "furious": "furious1", "annoyed": "irritated1", "irritated": "irritated1",
    "displeased": "displeased1", "contempt": "contempt1",
    "bored": "boredom1", "boredom": "boredom1",
    "indifferent": "indifferent1", "uncomfortable": "uncomfortable1",
    "relief": "relief1", "serenity": "serenity1", "calm": "calming1",
    "enthusiastic": "enthusiastic1", "energetic": "enthusiastic1",
    "attentive": "attentive1", "listening": "attentive1",
    "helpful": "helpful1", "understand": "understanding1",
    "inquiring": "inquiring1", "asking": "inquiring1",
    "incomprehensible": "incomprehensible2", "lost": "lost1",
    "oops": "oops1", "mistake": "oops1", "reprimand": "reprimand1",
    "impatient": "impatient1", "resigned": "resigned1",
    "go_away": "go_away1", "come": "come1",
    "electric": "electric1", "dying": "dying1", "die": "dying1",
}


def _tool_play_emotion(name: str = "", **_kwargs) -> dict:
    """Play a pre-recorded animation (emotion OR dance) from official HF libraries.

    Routing:
      1. name in dances library (19 names: yeah_nod / chicken_peck / chin_lead / ...)
         → play directly from pollen-robotics/reachy-mini-dances-library
      2. name == "dance" / "dancing" / "boogie" / "groove" → default dance "yeah_nod"
      3. name in alias map (happy / cheerful / sad / curious / proud / loving ...)
         → translate to clip and play from emotions library
      4. name matches an emotions-library clip exactly (e.g. dance3, cheerful2)
         → play directly
      5. otherwise → error

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

    # 1. Direct dance-library hit (dynamic — covers any name pollen publishes)
    if n in dances:
        ds = _urlparse.quote(_DANCES_LIB, safe="")
        return _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{n}")

    # 2. Generic "dance" intent → first dance in library (alphabetical, deterministic)
    if n in {"dance", "dancing", "boogie", "groove", "jig"}:
        pick = "yeah_nod" if "yeah_nod" in dances else (sorted(dances)[0] if dances else "yeah_nod")
        ds = _urlparse.quote(_DANCES_LIB, safe="")
        return _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{pick}")

    # 3. Direct emotion-library hit (dynamic — covers ALL clips, e.g. dance3,
    #    enthusiastic2, proud3, yes_sad1, oops2, understanding2 — beyond aliases)
    if n in emotions:
        ds = _urlparse.quote(_EMOTIONS_LIB, safe="")
        return _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{n}")

    # 4. English-friendly alias (LLM emits "happy" → cheerful1, etc.)
    if n in _EMOTION_ALIAS:
        clip = _EMOTION_ALIAS[n]
        ds = _urlparse.quote(_EMOTIONS_LIB, safe="")
        return _post_daemon(f"/api/move/play/recorded-move-dataset/{ds}/{clip}")

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
              "Move the robot head to a pose in degrees. All axes optional, default 0. Safe range ±25°.",
              {
                  "pitch": {"type": "number", "description": "Head tilt up/down in degrees (+down)."},
                  "yaw":   {"type": "number", "description": "Head turn left/right in degrees (+right)."},
                  "roll":  {"type": "number", "description": "Head tilt sideways in degrees."},
              }),
        _tool_move_head,
    ),
    "play_emotion": (
        _spec("play_emotion",
              "Play a pre-recorded animation: emotion OR dance. "
              "Dances: dance | yeah_nod | jackson_square | dizzy_spin | groovy_sway_and_roll | side_to_side_sway | pendulum_swing | grid_snap | chicken_peck | head_tilt_roll | uh_huh_tilt | side_peekaboo. "
              "Emotions: happy | sad | curious | thoughtful | greet | nod | shake | surprised | scared | proud | loving | confused | shy | tired | enthusiastic | grateful | bored | frustrated | angry | calm | welcome | listening. "
              "Use 'dance' for a generic dance, or a specific dance/emotion name.",
              {"name": {"type": "string", "description": "Animation name — 'dance' for generic dance, or a specific name from the lists above."}},
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
              "Look through the robot's camera and describe what is currently visible. "
              "Use this when the user asks 'what do you see', 'what's in front of you', "
              "or wants a general visual update. Optional query narrows focus.",
              {"query": {"type": "string", "description": "Optional focused question about the scene (e.g. 'what color is the mug')."}}),
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
