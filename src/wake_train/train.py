"""Classifier head training on openWakeWord 16-frame windows.

Pipeline:
  positive wavs  ->  pad to 2.0 s  ->  AudioFeatures.embed_clips  ->  (N_pos, 16, 96)
  HF neg shards (N, 96 per-frame stream)  ->  random 16-frame windows  ->  (N_neg, 16, 96)

Classifier architecture matches the bundled v0.1 heads (alexa.onnx etc.):
  input (batch, 16, 96)  ->  Flatten  ->  Linear(1536, hidden)  ->  ReLU  ->
  Linear(hidden, 1).  Training uses BCEWithLogitsLoss; export wraps with sigmoid.

ONNX I/O contract is identical to the bundled heads so the Phase B1 baseline
harness can load it via wakeword_model_paths=[onnx_path] with no other change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import WakeConfig

log = logging.getLogger(__name__)

FRAME_STACK = 16        # frames per inference window — must match bundled heads
FRAME_MS = 80           # ms per embedding frame at 16 kHz
WINDOW_S = FRAME_STACK * FRAME_MS / 1000.0  # 1.28 s
PAD_S = 2.0             # pad each positive wav to this duration before encode


@dataclass
class TrainResult:
    weights_path: Path
    meta_path: Path
    val_loss: float
    val_recall: float
    val_far_per_hr: float
    steps: int


def _load_wav_padded(wav_path: Path, sample_rate: int = 16_000) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr != sample_rate:
        raise ValueError(f"{wav_path} sr={sr} != {sample_rate}")
    target = int(round(PAD_S * sample_rate))
    if wav.size >= target:
        return wav[:target]
    out = np.zeros(target, dtype=np.int16)
    head = (target - wav.size) // 2
    out[head:head + wav.size] = wav
    return out


def _positive_windows(cfg: WakeConfig, manifest: dict) -> np.ndarray:
    """Encode every positive wav into 16-frame windows.

    embed_clips returns (n_clips, 16, 96) directly when each clip is exactly
    2.0 s long, so this is a straightforward batched call.
    """
    from openwakeword.utils import AudioFeatures

    paths = [cfg.data_dir / rec["wav"] for rec in manifest["positives"]
             if rec["slug"] in cfg.phrases]
    if not paths:
        raise RuntimeError("no positive wavs in manifest — synth + manifest step missing?")
    clips = np.stack([_load_wav_padded(p) for p in paths])
    log.info("encoding %d positive clips to embeddings", len(clips))
    af = AudioFeatures()
    feats = af.embed_clips(clips, batch_size=128)
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 3 or feats.shape[1] != FRAME_STACK or feats.shape[2] != 96:
        raise RuntimeError(f"unexpected embed_clips shape {feats.shape}; "
                           f"expected (N, {FRAME_STACK}, 96)")
    return feats


def _negative_windows(cfg: WakeConfig, n_windows: int) -> np.ndarray:
    """Sample 16-frame windows from precomputed per-frame neg shards."""
    paths = sorted(cfg.neg_dir.glob("negative_features_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no neg shards under {cfg.neg_dir}")

    rng = np.random.default_rng(cfg.train.seed)
    pooled: list[np.ndarray] = []
    for p in paths:
        a = np.load(p, mmap_mode="r")
        # accept either per-frame (N, 96) stream or pre-windowed (N, 16, 96)
        if a.ndim == 2 and a.shape[1] == 96:
            pooled.append(a)
        elif a.ndim == 3 and a.shape[1:] == (FRAME_STACK, 96):
            # already windowed — flatten back to frame stream (lose window
            # boundaries but recover uniform sampling)
            pooled.append(a.reshape(-1, 96))
        else:
            raise RuntimeError(f"unexpected neg shape in {p.name}: {a.shape}")

    # Sample windows lazily across all shards
    shard_sizes = [p.shape[0] for p in pooled]
    total = sum(shard_sizes)
    if total < FRAME_STACK + 1:
        raise RuntimeError(f"only {total} neg frames available; need at least 17")
    cum = np.cumsum(shard_sizes)

    out = np.empty((n_windows, FRAME_STACK, 96), dtype=np.float32)
    for i in range(n_windows):
        start = int(rng.integers(0, total - FRAME_STACK))
        shard_idx = int(np.searchsorted(cum, start, side="right"))
        local_start = start - (cum[shard_idx - 1] if shard_idx > 0 else 0)
        shard = pooled[shard_idx]
        if local_start + FRAME_STACK <= shard.shape[0]:
            out[i] = shard[local_start:local_start + FRAME_STACK]
        else:
            # crossing shard boundary — wrap to next shard (rare but safe)
            head = shard[local_start:]
            need = FRAME_STACK - head.shape[0]
            nxt = pooled[(shard_idx + 1) % len(pooled)][:need]
            out[i] = np.concatenate([head, nxt], axis=0)
    return out


def _build_classifier(hidden_dim: int, n_layers: int, dropout: float):
    """Flatten + MLP, matching bundled v0.1 head architecture."""
    import torch.nn as nn

    flat_dim = FRAME_STACK * 96
    layers: list[nn.Module] = [nn.Flatten()]
    in_dim = flat_dim
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
    pos = _positive_windows(cfg, manifest)
    log.info("positive windows: %s", pos.shape)

    n_pos_total = pos.shape[0]
    # 30x neg windows keeps a sane class ratio; cap by available data
    n_neg_total = n_pos_total * 30
    neg = _negative_windows(cfg, n_neg_total)
    log.info("negative windows: %s", neg.shape)

    rng = np.random.default_rng(cfg.train.seed)
    pos_idx = rng.permutation(n_pos_total)
    neg_idx = rng.permutation(neg.shape[0])
    pos_val_n = max(1, int(n_pos_total * cfg.train.val_split))
    neg_val_n = max(1, int(neg.shape[0] * cfg.train.val_split))
    pos_train, pos_val = pos[pos_idx[pos_val_n:]], pos[pos_idx[:pos_val_n]]
    neg_train, neg_val = neg[neg_idx[neg_val_n:]], neg[neg_idx[:neg_val_n]]

    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _build_classifier(
        cfg.train.hidden_dim, cfg.train.n_layers, cfg.train.dropout
    ).to(device)
    opt = AdamW(head.parameters(), lr=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def _batch(rng_local: np.random.Generator):
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
    head.train()
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
        all_x = torch.from_numpy(
            np.concatenate([pos_val, neg_val]).astype(np.float32)
        ).to(device)
        all_y = torch.from_numpy(
            np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))]).astype(np.float32)
        ).to(device).unsqueeze(1)
        val_loss = float(loss_fn(head(all_x), all_y))

    threshold = 0.5
    recall = float((sp >= threshold).mean()) if sp.size else 0.0
    far_rate = float((sn >= threshold).mean()) if sn.size else 0.0
    # 80 ms / frame -> 12.5 frames / s -> 45 000 frames / hour (window-level FAR)
    far_per_hr = far_rate * 45_000.0

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cfg.artifacts_dir / f"{cfg.name}.head.pt"
    meta_path = cfg.artifacts_dir / f"{cfg.name}.head.json"
    torch.save({
        "state_dict": head.state_dict(),
        "hidden_dim": cfg.train.hidden_dim,
        "n_layers": cfg.train.n_layers,
        "frame_stack": FRAME_STACK,
        "embedding_dim": 96,
    }, weights_path)
    meta_path.write_text(json.dumps({
        "name": cfg.name,
        "scope": cfg.scope,
        "frame_stack": FRAME_STACK,
        "embedding_dim": 96,
        "hidden_dim": cfg.train.hidden_dim,
        "n_layers": cfg.train.n_layers,
        "n_steps": cfg.train.n_steps,
        "n_pos_windows": int(n_pos_total),
        "n_neg_windows": int(neg.shape[0]),
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
