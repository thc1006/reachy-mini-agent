"""ONNX-runtime latency probe — drives the exported head with random
16-frame embedding windows. Mirrors _oww_baseline.py's measurement style.

Authoritative Pi 4 numbers come from copying the ONNX to the robot and
running scripts/wake_train/verify_pi.sh against Phase B1's baseline harness;
this x86 probe is a quick sanity check that no regression slipped into the
trained head.
"""

from __future__ import annotations

import logging
import statistics
import time

import numpy as np

from ..config import WakeConfig
from ..export import EMBEDDING_DIM
from ..train import FRAME_STACK

log = logging.getLogger(__name__)


def measure(cfg: WakeConfig, n_frames: int = 1000) -> dict:
    import onnxruntime as ort

    if not cfg.onnx_path.exists():
        raise FileNotFoundError(f"missing onnx: {cfg.onnx_path}")
    sess = ort.InferenceSession(str(cfg.onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    rng = np.random.default_rng(123)
    sample = rng.standard_normal((1, FRAME_STACK, EMBEDDING_DIM)).astype(np.float32)

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
