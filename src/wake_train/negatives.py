"""Fetch precomputed negative embeddings from openWakeWord.

openWakeWord publishes two negative-feature files on HuggingFace at
``davidscripka/openwakeword_features``:

  * ``validation_set_features.npy`` (180 MB) — a smaller, clean negative
    corpus normally used to compute FAR on a held-out validation set.
    Repurposing it for POC training keeps disk under 1 GB.
  * ``openwakeword_features_ACAV100M_2000_hrs_16bit.npy`` (17.28 GB) — the
    full 2000-hour training feature set, stored as int16 to save space.
    Used by PROD; dequantise to float32 at load time.

The pre-2025 multi-shard ``negative_features_*.npy`` layout this module
originally assumed does not exist; this is the authoritative source per
the openwakeword.com training docs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


NEG_REPO_ID = "davidscripka/openwakeword_features"
NEG_REPO_TYPE = "dataset"


@dataclass(frozen=True)
class NegativeSet:
    root: Path
    files: tuple[Path, ...]

    @property
    def n_files(self) -> int:
        return len(self.files)


def fetch(target_dir: Path, filenames: tuple[str, ...]) -> NegativeSet:
    """Download each ``filenames`` entry into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    paths: list[Path] = []
    for name in filenames:
        log.info("downloading %s from %s", name, NEG_REPO_ID)
        local = hf_hub_download(
            repo_id=NEG_REPO_ID,
            filename=name,
            repo_type=NEG_REPO_TYPE,
            local_dir=str(target_dir),
        )
        paths.append(Path(local))
    return NegativeSet(root=target_dir, files=tuple(paths))


def scan(target_dir: Path, filenames: tuple[str, ...] | None = None) -> NegativeSet:
    """Return the negative set already cached on disk, no network."""
    target_dir = Path(target_dir)
    if filenames is None:
        paths = tuple(sorted(target_dir.glob("*.npy")))
    else:
        paths = tuple(target_dir / f for f in filenames if (target_dir / f).exists())
    return NegativeSet(root=target_dir, files=paths)
