"""Augmentation primitives — pure numpy, no piper / openwakeword needed."""

from __future__ import annotations

import numpy as np
import pytest

from wake_train import augment
from wake_train.config import AugmentConfig


@pytest.fixture
def tone() -> np.ndarray:
    # 0.5 s, 440 Hz sine at 16 kHz, amplitude 0.5
    t = np.arange(8000) / 16000.0
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_add_white_noise_changes_signal(tone):
    rng = np.random.default_rng(0)
    noisy = augment.add_white_noise(tone, snr_db=10.0, rng=rng)
    assert noisy.shape == tone.shape
    assert noisy.dtype == np.float32
    assert not np.allclose(noisy, tone)


def test_add_white_noise_snr_roughly_correct(tone):
    rng = np.random.default_rng(0)
    snr_target = 10.0
    noisy = augment.add_white_noise(tone, snr_db=snr_target, rng=rng)
    sig_rms = augment._rms(tone)
    noise_rms = augment._rms(noisy - tone)
    measured = 20 * np.log10(sig_rms / max(noise_rms, 1e-9))
    assert abs(measured - snr_target) < 1.0


def test_random_gain_scales_signal(tone):
    louder = augment.random_gain(tone, db=6.0)
    quieter = augment.random_gain(tone, db=-6.0)
    assert np.max(np.abs(louder)) > np.max(np.abs(tone)) * 0.9
    assert np.max(np.abs(quieter)) < np.max(np.abs(tone))


def test_random_gain_avoids_clip():
    loud = np.full(1000, 0.9, dtype=np.float32)
    out = augment.random_gain(loud, db=12.0)
    assert float(np.max(np.abs(out))) <= 0.99 + 1e-6


def test_time_shift_preserves_length(tone):
    out = augment.time_shift(tone, shift_samples=200)
    assert out.shape == tone.shape
    assert float(out[0]) == 0.0
    out_neg = augment.time_shift(tone, shift_samples=-200)
    assert out_neg.shape == tone.shape


def test_pitch_shift_preserves_length(tone):
    out = augment.pitch_shift_resample(tone, semitones=2.0)
    assert out.shape == tone.shape
    assert out.dtype == np.float32


def test_mix_at_snr_handles_silent_signal():
    silent = np.zeros(1000, dtype=np.float32)
    interferer = np.ones(1000, dtype=np.float32)
    out = augment.mix_at_snr(silent, interferer, snr_db=0.0)
    # Silent target -> we pass through unchanged rather than dividing by zero
    assert out.shape == silent.shape


def test_pipeline_deterministic_with_seed(tone):
    cfg = AugmentConfig(seed=99)
    p1 = augment.AugPipeline(cfg)
    p2 = augment.AugPipeline(cfg)
    out1 = p1.apply(tone)
    out2 = p2.apply(tone)
    np.testing.assert_array_equal(out1, out2)


def test_pipeline_changes_signal(tone):
    cfg = AugmentConfig(seed=7)
    out = augment.AugPipeline(cfg).apply(tone)
    assert out.shape == tone.shape
    assert not np.allclose(out, tone)


def test_pipeline_disabled_passthrough(tone):
    cfg = AugmentConfig(enable=False)
    out = augment.AugPipeline(cfg).apply(tone)
    np.testing.assert_array_equal(out, tone)
