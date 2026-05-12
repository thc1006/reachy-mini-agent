"""ONNX runtime latency probe — mirrors _oww_baseline.py.

Drives the exported head with random embedding-shaped windows and reports
percentile per-frame latency. Use on the 5090 (x86) for a quick sanity check;
the authoritative Pi 4 number comes from copying the ONNX to the robot and
re-running _oww_baseline.py with the new path.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np

from ..config import WakeConfig
from ..export import OWW_FRAME_STACK

log = logging.getLogger(__name__)


def measure(cfg: WakeConfig, n_frames: int = 1000) -> dict:
    import onnxruntime as ort

    if not cfg.onnx_path.exists():
        raise FileNotFoundError(f"missing onnx: {cfg.onnx_path}")
    sess = ort.InferenceSession(str(cfg.onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    if shape[0] in (None, "batch", -1):
        shape[0] = 1
    rng = np.random.default_rng(123)
    sample = rng.standard_normal(tuple(shape)).astype(np.float32)

    # warmup
    for _ in range(20):
        sess.run(None, {inp.name: sample})

    lats: list[float] = []
    for _ in range(n_frames):
        t = time.perf_counter()
        sess.run(None, {inp.name: sample})
        lats.append((time.perf_counter() - t) * 1000.0)

    result = {
        "onnx_path": str(cfg.onnx_path),
        "n_frames": n_frames,
        "frame_ms": cfg.train.frame_ms,
        "mean_ms": round(statistics.mean(lats), 3),
        "p50_ms": round(statistics.median(lats), 3),
        "p95_ms": round(sorted(lats)[int(n_frames * 0.95)], 3),
        "p99_ms": round(sorted(lats)[int(n_frames * 0.99)], 3),
        "max_ms": round(max(lats), 3),
    }
    log.info("latency: %s", result)
    return result
