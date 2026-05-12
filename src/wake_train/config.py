"""POC and PROD training profiles.

A profile fully describes a training run: phrases, voices, dataset sizes,
augmentation knobs, training hyperparameters, and output paths. Profiles are
pure data so they can be unit-tested without any heavy dep installed.

POC budget targets (per user spec):
  * 2 Piper voices x ~5 phrase variants x sampling jitter -> ~2000 positives
  * augmentation pass adds 1x-2x variety on top
  * negatives: 1-2 precomputed feature shards from dscripka/openwakeword_features
  * training steps capped so total wall time on a single 5090 ~= 2 h
  * disk footprint ~= 3-5 GB

PROD additions are gated behind a passing POC eval and switch to the full
recipe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Scope = Literal["poc", "prod"]


@dataclass(frozen=True)
class AugmentConfig:
    enable: bool = True
    noise_prob: float = 0.7
    noise_snr_db_range: tuple[float, float] = (0.0, 25.0)
    gain_prob: float = 0.8
    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    pitch_prob: float = 0.5
    pitch_semitones_range: tuple[float, float] = (-2.0, 2.0)
    time_shift_prob: float = 0.5
    time_shift_ms_range: tuple[float, float] = (-120.0, 120.0)
    rir_prob: float = 0.3
    background_prob: float = 0.4
    background_snr_db_range: tuple[float, float] = (5.0, 20.0)
    seed: int = 1337


@dataclass(frozen=True)
class TrainConfig:
    sample_rate: int = 16_000
    frame_ms: int = 80
    embedding_dim: int = 96
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.2
    batch_size: int = 1024
    learning_rate: float = 1e-3
    n_steps: int = 50_000
    warmup_steps: int = 1_000
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class WakeConfig:
    scope: Scope
    name: str
    phrases: tuple[str, ...]
    voices: tuple[str, ...]
    n_synth_per_voice_phrase: int
    real_recordings_dir: Path | None
    negative_files: tuple[str, ...]
    augment: AugmentConfig
    train: TrainConfig
    base_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data" / "wake_train" / self.scope

    @property
    def synth_dir(self) -> Path:
        return self.data_dir / "synth"

    @property
    def aug_dir(self) -> Path:
        return self.data_dir / "aug"

    @property
    def neg_dir(self) -> Path:
        return self.data_dir / "neg"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / "artifacts" / "wake" / self.scope

    @property
    def onnx_path(self) -> Path:
        return self.artifacts_dir / f"{self.name}.onnx"

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / f"{self.name}.report.json"


# Wake phrase set (see phrases.py for slug mapping)
_PHRASES_POC: tuple[str, ...] = (
    "hei_ruiqi",        # 嘿瑞奇
    "hey_reachy",       # Hey Reachy
)
_PHRASES_PROD: tuple[str, ...] = (
    "hei_ruiqi",        # 嘿瑞奇
    "hey_reachy",       # Hey Reachy
    "ruiqi",            # 瑞奇
    "hei_ruiqi_polite", # 嘿，瑞奇。
    "reachy",           # Reachy
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


_BASE_DIR = Path(os.environ.get("WAKE_TRAIN_BASE_DIR", str(_repo_root())))


POC_CONFIG = WakeConfig(
    scope="poc",
    name="hei_ruiqi_poc_v0",
    phrases=_PHRASES_POC,
    voices=(
        "zh_CN-huayan-medium",
        "en_US-amy-medium",
    ),
    n_synth_per_voice_phrase=200,   # 2 voices x 2 phrases x 200 = 800 raw,
                                    # aug expands ~3x -> ~2.4k positives
    real_recordings_dir=None,
    # POC uses the 180 MB validation corpus repurposed as training negatives;
    # full 17 GB ACAV100M file is PROD only.
    negative_files=("validation_set_features.npy",),
    augment=AugmentConfig(),
    train=TrainConfig(),
    base_dir=_BASE_DIR,
)


PROD_CONFIG = WakeConfig(
    scope="prod",
    name="hei_ruiqi_prod_v0",
    phrases=_PHRASES_PROD,
    voices=(
        "zh_CN-huayan-medium",
        "zh_CN-huayan-x_low",
        "en_US-amy-medium",
        "en_US-ryan-medium",
        "en_US-lessac-medium",
        "en_GB-northern_english_male-medium",
        "fr_FR-siwis-medium",
        "de_DE-thorsten-medium",
    ),
    n_synth_per_voice_phrase=125,   # 8 voices x 5 phrases x 125 = 5000 raw
    real_recordings_dir=_BASE_DIR / "_wake_probe_real",
    # PROD uses the 17 GB ACAV100M 2000-hour feature file (int16 dequant -> f32)
    negative_files=(
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    ),
    augment=AugmentConfig(),
    train=TrainConfig(n_steps=200_000, warmup_steps=5_000),
    base_dir=_BASE_DIR,
)


CONFIGS: dict[Scope, WakeConfig] = {
    "poc": POC_CONFIG,
    "prod": PROD_CONFIG,
}


def get(scope: Scope) -> WakeConfig:
    if scope not in CONFIGS:
        raise ValueError(f"unknown scope: {scope!r} (expected one of {list(CONFIGS)})")
    return CONFIGS[scope]
