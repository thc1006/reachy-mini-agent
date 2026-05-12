"""Held-out recall + neg-corpus FAR on the exported ONNX head.

This evaluates the model exactly the way the runtime sees it: 16-frame
stacked embedding windows -> sigmoid score -> threshold. It re-uses the
train-time positive/negative split (seeded by cfg.train.seed) so the numbers
are reproducible after retraining.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import WakeConfig
from ..export import OWW_FRAME_STACK

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def _windowed(frames: np.ndarray) -> np.ndarray:
    """Slide an OWW_FRAME_STACK window across the frame axis."""
    if frames.shape[0] < OWW_FRAME_STACK:
        return np.zeros((0, 1, OWW_FRAME_STACK, frames.shape[1]), dtype=np.float32)
    n = frames.shape[0] - OWW_FRAME_STACK + 1
    out = np.zeros((n, 1, OWW_FRAME_STACK, frames.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i, 0] = frames[i:i + OWW_FRAME_STACK]
    return out


def _features_from_wav(wav_path: Path) -> np.ndarray:
    import soundfile as sf
    from openwakeword.utils import AudioFeatures

    wav, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr != 16_000:
        raise ValueError(f"{wav_path} sr={sr} != 16000")
    af = AudioFeatures()
    af(wav)
    return np.array(af._raw_embeddings, dtype=np.float32, copy=True)


def measure(cfg: WakeConfig) -> dict:
    import onnxruntime as ort

    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(cfg.onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(cfg.train.seed)
    pos_records = [r for r in manifest["positives"] if r["slug"] in cfg.phrases]
    rng.shuffle(pos_records)
    val_n = max(1, int(len(pos_records) * cfg.train.val_split))
    val_records = pos_records[:val_n]

    # Per-clip recall: a clip counts as detected if any window crosses threshold.
    pos_max_scores: list[float] = []
    for rec in val_records:
        wav_abs = cfg.data_dir / rec["wav"]
        try:
            emb = _features_from_wav(wav_abs)
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", wav_abs, e)
            continue
        windows = _windowed(emb)
        if windows.shape[0] == 0:
            continue
        scores = sess.run(None, {inp_name: windows})[0].squeeze()
        pos_max_scores.append(float(np.max(scores)))

    # Neg: sample windows from precomputed neg embedding shards.
    neg_paths = sorted(cfg.neg_dir.glob("negative_features_*.npy"))
    if not neg_paths:
        raise FileNotFoundError(f"no neg shards under {cfg.neg_dir}")
    neg_emb = np.load(neg_paths[0], mmap_mode="r")
    n_neg_windows = min(20_000, neg_emb.shape[0] - OWW_FRAME_STACK)
    starts = rng.integers(0, neg_emb.shape[0] - OWW_FRAME_STACK, size=n_neg_windows)
    BATCH = 1024
    neg_scores: list[np.ndarray] = []
    for i in range(0, len(starts), BATCH):
        chunk = np.stack(
            [neg_emb[s:s + OWW_FRAME_STACK] for s in starts[i:i + BATCH]]
        )[:, None, :, :].astype(np.float32)
        neg_scores.append(sess.run(None, {inp_name: chunk})[0].squeeze())
    neg_all = np.concatenate(neg_scores) if neg_scores else np.zeros(0)

    pos_arr = np.array(pos_max_scores, dtype=np.float32)
    by_threshold = {}
    for thr in DEFAULT_THRESHOLDS:
        recall = float((pos_arr >= thr).mean()) if pos_arr.size else 0.0
        far_rate = float((neg_all >= thr).mean()) if neg_all.size else 0.0
        by_threshold[f"{thr:.2f}"] = {
            "recall": recall,
            "far_rate": far_rate,
            "far_per_hour": far_rate * 45_000.0,
        }

    result = {
        "scope": cfg.scope,
        "name": cfg.name,
        "n_pos_clips": len(pos_arr),
        "n_neg_windows": int(neg_all.size),
        "thresholds": by_threshold,
    }
    log.info("accuracy: %s", result)
    return result
