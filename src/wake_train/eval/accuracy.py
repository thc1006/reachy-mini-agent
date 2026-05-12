"""Held-out recall + neg-corpus FAR on the exported ONNX head.

Reuses the train-time positive split (seeded by cfg.train.seed) so the
recall number is the same data the trainer didn't see. Negative windows
are randomly sampled from the precomputed feature shards, matching the
inference contract (16-frame windows / 96-dim embeddings).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from ..config import WakeConfig
from ..export import EMBEDDING_DIM
from ..train import FRAME_STACK, _load_wav_padded

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def _encode_clips(paths: list) -> np.ndarray:
    from openwakeword.utils import AudioFeatures

    if not paths:
        return np.zeros((0, FRAME_STACK, EMBEDDING_DIM), dtype=np.float32)
    clips = np.stack([_load_wav_padded(p) for p in paths])
    af = AudioFeatures()
    feats = np.asarray(af.embed_clips(clips, batch_size=128), dtype=np.float32)
    return feats


def measure(cfg: WakeConfig) -> dict:
    import onnxruntime as ort

    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(cfg.onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(cfg.train.seed)
    pos_records = [r for r in manifest["positives"] if r["slug"] in cfg.phrases]
    rng.shuffle(pos_records)
    val_n = max(1, int(len(pos_records) * cfg.train.val_split))
    val_paths = [cfg.data_dir / r["wav"] for r in pos_records[:val_n]]
    pos_feats = _encode_clips(val_paths)

    pos_scores = np.zeros(pos_feats.shape[0], dtype=np.float32)
    for i, window in enumerate(pos_feats):
        out = sess.run(None, {inp_name: window[None].astype(np.float32)})[0]
        pos_scores[i] = float(out[0, 0])

    # Negatives: sample windows from precomputed feature file(s)
    from ..train import _load_neg_file, _dequant_window

    neg_paths = [cfg.neg_dir / fn for fn in cfg.negative_files]
    missing = [p for p in neg_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing neg files in {cfg.neg_dir}: {[p.name for p in missing]}"
        )
    neg_emb = _load_neg_file(neg_paths[0])
    n_neg = min(20_000, neg_emb.shape[0] - FRAME_STACK)
    starts = rng.integers(0, neg_emb.shape[0] - FRAME_STACK, size=n_neg)
    neg_scores = np.zeros(n_neg, dtype=np.float32)
    for i, s in enumerate(starts):
        window = _dequant_window(neg_emb[s:s + FRAME_STACK])[None]
        neg_scores[i] = float(sess.run(None, {inp_name: window})[0][0, 0])

    by_threshold = {}
    for thr in DEFAULT_THRESHOLDS:
        recall = float((pos_scores >= thr).mean()) if pos_scores.size else 0.0
        far_rate = float((neg_scores >= thr).mean()) if neg_scores.size else 0.0
        by_threshold[f"{thr:.2f}"] = {
            "recall": recall,
            "far_rate": far_rate,
            "far_per_hour": far_rate * 45_000.0,
        }

    result = {
        "scope": cfg.scope,
        "name": cfg.name,
        "n_pos_clips": int(pos_scores.size),
        "n_neg_windows": int(neg_scores.size),
        "thresholds": by_threshold,
    }
    log.info("accuracy: %s", result)
    return result
