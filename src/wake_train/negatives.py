"""Fetch precomputed negative embeddings from openWakeWord.

The upstream training recipe relies on ~30k hours of negative audio
pre-encoded into the openWakeWord embedding space and published as a
HuggingFace dataset (dscripka/openwakeword_features). POC pulls a 2-shard
subset (~3 GB) so the classifier sees enough diversity to converge without
the full 24 GB cost.

This module is a thin wrapper around huggingface_hub.snapshot_download so the
training entrypoint can be deterministic about which shards it consumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


NEG_REPO_ID = "dscripka/openwakeword_features"
NEG_REPO_TYPE = "dataset"
NEG_FILE_PATTERN = "negative_features_*.npy"


@dataclass(frozen=True)
class NegativeSet:
    root: Path
    files: tuple[Path, ...]

    @property
    def n_files(self) -> int:
        return len(self.files)


def fetch(target_dir: Path, n_shards: int) -> NegativeSet:
    """Download ``n_shards`` of negative features into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    available = sorted(
        f for f in api.list_repo_files(NEG_REPO_ID, repo_type=NEG_REPO_TYPE)
        if f.startswith("negative_features_") and f.endswith(".npy")
    )
    if not available:
        raise RuntimeError(
            f"no negative feature shards found in {NEG_REPO_ID!r}; "
            "openWakeWord may have changed its dataset layout"
        )
    chosen = available[:n_shards]
    log.info("downloading %d/%d neg shards from %s", len(chosen), len(available), NEG_REPO_ID)
    paths: list[Path] = []
    for name in chosen:
        local = hf_hub_download(
            repo_id=NEG_REPO_ID,
            filename=name,
            repo_type=NEG_REPO_TYPE,
            local_dir=str(target_dir),
        )
        paths.append(Path(local))
    return NegativeSet(root=target_dir, files=tuple(paths))


def scan(target_dir: Path) -> NegativeSet:
    """Return the negative set already cached on disk, no network."""
    target_dir = Path(target_dir)
    paths = tuple(sorted(target_dir.glob(NEG_FILE_PATTERN)))
    return NegativeSet(root=target_dir, files=paths)
