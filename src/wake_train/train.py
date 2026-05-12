"""PyTorch classifier head training on openWakeWord embeddings.

openWakeWord splits its pipeline as: shared mel + embedding net (frozen,
shared across all wake heads) -> per-wakeword classifier (the only learned
part). We mirror that: feed positive wavs through openwakeword's AudioFeatures
to get (frames, embedding_dim) tensors, load matching-shape negative
embeddings from the published .npy shards, and train a small MLP head.

The trained head exports to ONNX with the same input contract as the bundled
heads (alexa.onnx etc.) so Phase B1's Pi 4 baseline harness can drop it in by
path without code changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import WakeConfig

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    weights_path: Path
    meta_path: Path
    val_loss: float
    val_recall: float
    val_far_per_hr: float
    steps: int


def _features_from_wav(wav_path: Path, audio_features) -> np.ndarray:
    """Encode a single wav into the openWakeWord embedding space."""
    import soundfile as sf

    wav, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr != 16_000:
        raise ValueError(f"{wav_path} sample rate {sr} != 16000")
    # AudioFeatures expects int16, exposes embedding buffer via _get_embeddings
    audio_features.reset()
    audio_features(wav)
    return np.array(audio_features._raw_embeddings, dtype=np.float32, copy=True)


def _positive_features(cfg: WakeConfig, manifest: dict, slug: str) -> np.ndarray:
    from openwakeword.utils import AudioFeatures  # lazy, heavy

    af = AudioFeatures()
    chunks: list[np.ndarray] = []
    for rec in manifest["positives"]:
        if rec["slug"] != slug:
            continue
        wav_abs = cfg.data_dir / rec["wav"]
        emb = _features_from_wav(wav_abs, af)
        # take all overlapping windows (frame dim) as positives
        if emb.size:
            chunks.append(emb)
    if not chunks:
        return np.zeros((0, cfg.train.embedding_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _load_negatives(cfg: WakeConfig, max_rows: int | None = None) -> np.ndarray:
    paths = sorted(cfg.neg_dir.glob("negative_features_*.npy"))
    arrs: list[np.ndarray] = []
    total = 0
    for p in paths:
        a = np.load(p, mmap_mode="r")
        arrs.append(np.array(a, dtype=np.float32))
        total += a.shape[0]
        if max_rows is not None and total >= max_rows:
            break
    if not arrs:
        raise FileNotFoundError(f"no neg shards under {cfg.neg_dir}")
    neg = np.concatenate(arrs, axis=0)
    if max_rows is not None and neg.shape[0] > max_rows:
        rng = np.random.default_rng(cfg.train.seed)
        idx = rng.choice(neg.shape[0], max_rows, replace=False)
        neg = neg[idx]
    return neg


def _build_head(embedding_dim: int, hidden_dim: int, n_layers: int, dropout: float):
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    in_dim = embedding_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def train(cfg: WakeConfig) -> TrainResult:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW

    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    # POC trains one head per phrase but writes them into a shared multi-head
    # ONNX at export time. For now we collapse all phrases into a single
    # binary classifier (wake-vs-not-wake) — keeps POC simple; multi-head
    # support lands in PROD when we have per-slug labelled real recordings.
    pos_chunks = [
        _positive_features(cfg, manifest, slug) for slug in cfg.phrases
    ]
    pos = np.concatenate(pos_chunks, axis=0) if pos_chunks else np.zeros((0, 0), dtype=np.float32)
    if pos.size == 0:
        raise RuntimeError("no positive embeddings — synth + manifest step missing?")
    log.info("positive rows: %d (dim=%d)", pos.shape[0], pos.shape[1])

    # cap negatives at 30x positives to keep class balance sane on POC
    max_neg = pos.shape[0] * 30
    neg = _load_negatives(cfg, max_rows=max_neg)
    log.info("negative rows: %d (dim=%d)", neg.shape[0], neg.shape[1])
    if neg.shape[1] != pos.shape[1]:
        raise ValueError(
            f"embedding dim mismatch: pos={pos.shape[1]} neg={neg.shape[1]}"
        )

    rng = np.random.default_rng(cfg.train.seed)
    pos_idx = rng.permutation(pos.shape[0])
    neg_idx = rng.permutation(neg.shape[0])
    pos_val_n = max(1, int(pos.shape[0] * cfg.train.val_split))
    neg_val_n = max(1, int(neg.shape[0] * cfg.train.val_split))

    pos_train, pos_val = pos[pos_idx[pos_val_n:]], pos[pos_idx[:pos_val_n]]
    neg_train, neg_val = neg[neg_idx[neg_val_n:]], neg[neg_idx[:neg_val_n]]

    embedding_dim = pos.shape[1]
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _build_head(
        embedding_dim, cfg.train.hidden_dim, cfg.train.n_layers, cfg.train.dropout
    ).to(device)
    opt = AdamW(head.parameters(), lr=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def _batch(rng_local: np.random.Generator) -> tuple["torch.Tensor", "torch.Tensor"]:
        half = cfg.train.batch_size // 2
        p_idx = rng_local.integers(0, pos_train.shape[0], size=half)
        n_idx = rng_local.integers(0, neg_train.shape[0], size=half)
        x = np.concatenate([pos_train[p_idx], neg_train[n_idx]], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(half, dtype=np.float32),
                            np.zeros(half, dtype=np.float32)])
        return (torch.from_numpy(x).to(device),
                torch.from_numpy(y).to(device).unsqueeze(1))

    log.info("training %d steps on %s", cfg.train.n_steps, device)
    rng_local = np.random.default_rng(cfg.train.seed + 1)
    for step in range(1, cfg.train.n_steps + 1):
        x, y = _batch(rng_local)
        logits = head(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, cfg.train.n_steps // 20) == 0:
            log.info("step %d/%d  loss=%.4f", step, cfg.train.n_steps, float(loss))

    head.eval()
    with torch.no_grad():
        xp = torch.from_numpy(pos_val.astype(np.float32)).to(device)
        xn = torch.from_numpy(neg_val.astype(np.float32)).to(device)
        sp = torch.sigmoid(head(xp)).cpu().numpy().squeeze(-1)
        sn = torch.sigmoid(head(xn)).cpu().numpy().squeeze(-1)
    threshold = 0.5
    recall = float((sp >= threshold).mean()) if sp.size else 0.0
    # frame-level FAR -> per-hour: each frame = 80 ms -> 45000 frames/hour
    far_rate = float((sn >= threshold).mean()) if sn.size else 0.0
    far_per_hr = far_rate * 45_000.0
    val_loss = float(
        loss_fn(
            head(torch.from_numpy(np.concatenate([pos_val, neg_val]).astype(np.float32)).to(device)),
            torch.from_numpy(
                np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))]).astype(np.float32)
            ).to(device).unsqueeze(1),
        )
    )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cfg.artifacts_dir / f"{cfg.name}.head.pt"
    meta_path = cfg.artifacts_dir / f"{cfg.name}.head.json"
    torch.save({"state_dict": head.state_dict(),
                "embedding_dim": embedding_dim,
                "hidden_dim": cfg.train.hidden_dim,
                "n_layers": cfg.train.n_layers}, weights_path)
    meta_path.write_text(json.dumps({
        "name": cfg.name,
        "scope": cfg.scope,
        "embedding_dim": embedding_dim,
        "hidden_dim": cfg.train.hidden_dim,
        "n_layers": cfg.train.n_layers,
        "n_steps": cfg.train.n_steps,
        "val_recall": recall,
        "val_far_per_hr": far_per_hr,
        "val_loss": val_loss,
    }, indent=2), encoding="utf-8")

    log.info("val recall=%.3f  FAR/hr=%.2f  loss=%.4f", recall, far_per_hr, val_loss)
    return TrainResult(
        weights_path=weights_path,
        meta_path=meta_path,
        val_loss=val_loss,
        val_recall=recall,
        val_far_per_hr=far_per_hr,
        steps=cfg.train.n_steps,
    )
