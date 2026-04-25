"""Cascaded perception · Tier 1 (always-on YOLO object detector).

Goal: run a cheap (~10ms) detector continuously so the LLM has fresh
"objects in the room" context for free, and only invoke the heavy VLM
when the detection list changes meaningfully or the LLM explicitly asks.

This module is opt-in via env `VISION_CASCADE=1`. Default off.

Public API:
    start_cascade(stop_event, frame_getter)
        Spawns a background thread.
    get_objects() -> list[str]
        Latest unique object names. Cheap, cache-only.
    objects_changed() -> bool
        Whether the set has changed since the last VLM trigger. Resetting it
        is the caller's responsibility via mark_objects_seen().
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional

# YOLOv8n is the smallest practical detector — ~6 MB, ~3-15 ms per 640x480
# frame on a 3090. Falls back to CPU at ~30-60 ms which is still fine for
# a 2-Hz cascade.
_MODEL_NAME = os.getenv("CASCADE_MODEL", "yolov8n.pt")
_INTERVAL_S = float(os.getenv("CASCADE_INTERVAL", "0.5"))   # 2 Hz default
_CONF_THRESH = float(os.getenv("CASCADE_CONF", "0.35"))
_TOPK = int(os.getenv("CASCADE_TOPK", "8"))

_state_lock = threading.Lock()
_objects: List[str] = []
_objects_t: float = 0.0
_last_trigger_set: frozenset = frozenset()


def _maybe_log(msg: str):
    if os.getenv("CASCADE_VERBOSE", "0") == "1":
        print(msg, flush=True)


def _load_model():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"  [cascade] ultralytics not installed: {e}; cascade disabled")
        return None
    try:
        return YOLO(_MODEL_NAME)
    except Exception as e:
        print(f"  [cascade] YOLO load failed ({_MODEL_NAME}): {e}")
        return None


def _detect(model, frame) -> List[str]:
    if model is None or frame is None:
        return []
    try:
        # ultralytics auto-handles BGR/RGB and resize
        results = model.predict(frame, conf=_CONF_THRESH, verbose=False, imgsz=320)
        if not results:
            return []
        names = results[0].names                          # {idx: "person", ...}
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None:
            return []
        ids = boxes.cls.cpu().numpy().astype(int).tolist()
        seen: list[str] = []
        for i in ids:
            n = names.get(i, str(i))
            if n not in seen:
                seen.append(n)
            if len(seen) >= _TOPK:
                break
        return seen
    except Exception as e:
        _maybe_log(f"  [cascade] detect err: {e}")
        return []


def _worker(stop_event: threading.Event, frame_getter: Callable[[], object]):
    print(f"  [cascade] start ({_MODEL_NAME}, every {_INTERVAL_S}s)")
    model = _load_model()
    if model is None:
        return
    global _objects, _objects_t
    while not stop_event.is_set():
        time.sleep(_INTERVAL_S)
        frame = frame_getter()
        if frame is None:
            continue
        det = _detect(model, frame)
        with _state_lock:
            _objects = det
            _objects_t = time.time()
        _maybe_log(f"  [cascade] {det}")
    print("  [cascade] stop")


def start_cascade(stop_event: threading.Event, frame_getter: Callable[[], object]) -> Optional[threading.Thread]:
    """Start the cascade thread. Returns the Thread or None if disabled."""
    if os.getenv("VISION_CASCADE", "0") != "1":
        return None
    t = threading.Thread(
        target=_worker, args=(stop_event, frame_getter), daemon=True, name="cascade"
    )
    t.start()
    return t


def get_objects() -> List[str]:
    """Return the latest detected object list. Empty list if cascade not running
    or detection is stale (>3 s)."""
    with _state_lock:
        if _objects_t and time.time() - _objects_t < 3.0:
            return list(_objects)
    return []


def objects_changed() -> bool:
    """Whether the set has changed vs the last time mark_objects_seen() was
    called. Used by the smart-trigger logic to decide when to invoke the
    expensive VLM."""
    global _last_trigger_set
    with _state_lock:
        cur = frozenset(_objects)
    return cur != _last_trigger_set


def mark_objects_seen():
    global _last_trigger_set
    with _state_lock:
        _last_trigger_set = frozenset(_objects)


def objects_summary() -> str:
    """Compact string for LLM system prompt injection."""
    objs = get_objects()
    if not objs:
        return ""
    return "Objects currently visible: " + ", ".join(objs) + "."
