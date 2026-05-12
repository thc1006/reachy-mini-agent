"""POC vs PROD profile invariants."""

from __future__ import annotations

import pytest

from wake_train import config as cfgmod


def test_get_returns_poc_and_prod():
    assert cfgmod.get("poc").scope == "poc"
    assert cfgmod.get("prod").scope == "prod"


def test_get_rejects_unknown():
    with pytest.raises(ValueError, match="unknown scope"):
        cfgmod.get("bogus")  # type: ignore[arg-type]


def test_poc_is_smaller_than_prod():
    poc, prod = cfgmod.POC_CONFIG, cfgmod.PROD_CONFIG
    assert len(poc.voices) < len(prod.voices)
    assert len(poc.phrases) <= len(prod.phrases)
    assert poc.negative_files != prod.negative_files
    assert any("validation" in f for f in poc.negative_files)
    assert any("ACAV100M" in f or "2000_hrs" in f for f in prod.negative_files)
    assert poc.train.n_steps < prod.train.n_steps


def test_poc_synth_budget_in_spec():
    """User spec: POC ~= 2k synth positives."""
    cfg = cfgmod.POC_CONFIG
    total = len(cfg.voices) * len(cfg.phrases) * cfg.n_synth_per_voice_phrase
    # 2 voices * 2 phrases * 200 = 800 raw synth; aug stage multiplies later
    assert 500 <= total <= 1500


def test_artifact_paths_consistent():
    cfg = cfgmod.POC_CONFIG
    assert cfg.onnx_path.suffix == ".onnx"
    assert cfg.onnx_path.name.startswith(cfg.name)
    assert cfg.data_dir.is_absolute()
    assert cfg.artifacts_dir.is_absolute()


def test_real_recordings_only_for_prod():
    assert cfgmod.POC_CONFIG.real_recordings_dir is None
    assert cfgmod.PROD_CONFIG.real_recordings_dir is not None
