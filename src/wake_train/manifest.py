"""Build a unified training manifest.

The manifest lists every positive wav (synth + augmented + real) with its
phrase slug. Negatives stay as precomputed .npy embeddings and are referenced
by feature-file path, not enumerated per-frame.

Manifest shape:
    {
      "version": 1,
      "scope": "poc",
      "phrases": ["hei_ruiqi", "hey_reachy"],
      "sample_rate": 16000,
      "positives": [{"wav": "...", "slug": "hei_ruiqi", "source": "synth|aug|real"}, ...],
      "negatives": [{"path": "neg/negative_features_00.npy"}, ...]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import WakeConfig
from .negatives import scan as scan_negatives

log = logging.getLogger(__name__)


def _collect_wavs(root: Path, source: str, allowed_slugs: set[str]) -> list[dict]:
    if not root.exists():
        return []
    out: list[dict] = []
    for wav in sorted(root.rglob("*.wav")):
        # path layout produced by synth: <voice>/<slug>/<seed>.wav
        try:
            slug = wav.parent.name
        except IndexError:
            continue
        if slug not in allowed_slugs:
            continue
        out.append({
            "wav": str(wav.relative_to(root.parents[1])),
            "slug": slug,
            "source": source,
        })
    return out


def build(cfg: WakeConfig) -> Path:
    allowed = set(cfg.phrases)
    positives: list[dict] = []
    positives += _collect_wavs(cfg.synth_dir, "synth", allowed)
    positives += _collect_wavs(cfg.aug_dir, "aug", allowed)
    if cfg.real_recordings_dir is not None:
        positives += _collect_wavs(cfg.real_recordings_dir, "real", allowed)

    neg = scan_negatives(cfg.neg_dir)
    negatives = [{"path": str(p.relative_to(cfg.data_dir))} for p in neg.files]

    manifest = {
        "version": 1,
        "scope": cfg.scope,
        "name": cfg.name,
        "phrases": list(cfg.phrases),
        "voices": list(cfg.voices),
        "sample_rate": cfg.train.sample_rate,
        "positives": positives,
        "negatives": negatives,
    }
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "manifest: %d positives (%s), %d neg shards -> %s",
        len(positives),
        ", ".join(sorted({p["source"] for p in positives})) or "none",
        len(negatives),
        cfg.manifest_path,
    )
    return cfg.manifest_path
