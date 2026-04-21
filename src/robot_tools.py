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
                    **_kwargs) -> dict:
    """Move head to (pitch, yaw, roll) in degrees. Safe range: ±25° per axis.

    Values outside range are clamped silently (not an error — LLM may overshoot).
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
    # Call daemon (goal_head_pose-ish — daemon's move API accepts pose dicts)
    result = _post_daemon("/api/move/play/look_around", {"pitch": p, "yaw": y, "roll": r})
    if clipped:
        result["clipped"] = True
    return result


def _tool_play_emotion(name: str = "", **_kwargs) -> dict:
    """Play a pre-baked emotion animation: happy | sad | curious | think | greet."""
    name = (name or "").strip().lower()
    ALLOWED = {"happy", "sad", "curious", "think", "greet", "shake", "nod"}
    if name not in ALLOWED:
        return {"error": f"unknown emotion {name!r}, valid: {sorted(ALLOWED)}"}
    return _post_daemon(f"/api/move/play/{name}")


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
              "Play an emotion animation. Choose from: happy, sad, curious, think, greet, shake, nod.",
              {"name": {"type": "string", "description": "Emotion name."}},
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
