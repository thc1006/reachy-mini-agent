"""Lightweight thread-safe motor command logger.

Writes JSONL records to ``MOTOR_LOG_PATH`` env. No-op when unset.
Used to empirically diagnose motor race conditions, delta clamps, and
external command sync between face tracker and LLM tool_calls.

Each line is a JSON object with at least: ts, thread, event, plus
event-specific fields (pitch, yaw, body_yaw, source, clamped, ...).

Reader can ``jq`` the file to see exactly which thread commanded what
pose at what time, and whether the delta clamp engaged.
"""
import json
import os
import threading
import time

_PATH = os.getenv("MOTOR_LOG_PATH", "").strip()
_lock = threading.Lock()


def log_motor(event: str, **fields) -> None:
    """Append one record. Safe to call from any thread; no exception escapes."""
    if not _PATH:
        return
    rec = {
        "ts": round(time.time(), 4),
        "thread": threading.current_thread().name,
        "event": event,
        **fields,
    }
    try:
        line = json.dumps(rec, ensure_ascii=False, default=str)
    except Exception:
        return
    try:
        with _lock:
            with open(_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass
