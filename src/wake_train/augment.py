"""Numpy-only audio augmentation primitives.

Keeping the pipeline numpy-only means tests run on any host without librosa,
torchaudio, or piper installed. Heavier transforms (RIR convolution,
background mixing) read from disk only when invoked.

All functions are float32 in / float32 out, mono, at a fixed sample rate
(default 16 kHz). The pipeline is deterministic given a numpy `Generator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AugmentConfig

EPS = 1e-9


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + EPS))


def add_white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_rms = _rms(audio)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.standard_normal(audio.shape).astype(np.float32) * target_noise_rms
    return (audio + noise).astype(np.float32)


def random_gain(audio: np.ndarray, db: float) -> np.ndarray:
    scale = 10.0 ** (db / 20.0)
    out = audio * scale
    # avoid hard clip — soft-clip preserves training-time SNR estimation
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def time_shift(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    """Circular-pad shift. Negative shifts left, positive shifts right."""
    if shift_samples == 0 or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    out = np.zeros_like(audio)
    if shift_samples > 0:
        out[shift_samples:] = audio[: audio.size - shift_samples]
    else:
        s = -shift_samples
        out[: audio.size - s] = audio[s:]
    return out.astype(np.float32)


def pitch_shift_resample(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Cheap pitch shift via resampling + length restoration.

    Equivalent to playing the clip back at a different speed and then
    truncating/padding back to the original length. Pitch and duration are
    coupled; for POC this is fine since we also want some duration jitter.
    """
    if semitones == 0.0 or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    factor = 2.0 ** (semitones / 12.0)
    new_len = max(1, int(round(audio.size / factor)))
    x = np.linspace(0.0, audio.size - 1, new_len, dtype=np.float64)
    idx0 = np.floor(x).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, audio.size - 1)
    frac = (x - idx0).astype(np.float32)
    resampled = (1.0 - frac) * audio[idx0] + frac * audio[idx1]
    if resampled.size >= audio.size:
        return resampled[: audio.size].astype(np.float32)
    out = np.zeros_like(audio)
    out[: resampled.size] = resampled
    return out.astype(np.float32)


def mix_at_snr(
    audio: np.ndarray,
    interferer: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    if interferer.size == 0:
        return audio.astype(np.float32, copy=False)
    if interferer.size < audio.size:
        reps = int(np.ceil(audio.size / interferer.size))
        interferer = np.tile(interferer, reps)
    interferer = interferer[: audio.size]
    sig_rms = _rms(audio)
    if sig_rms < EPS:
        return audio.astype(np.float32, copy=False)
    noise_rms = _rms(interferer)
    if noise_rms < EPS:
        return audio.astype(np.float32, copy=False)
    scale = (sig_rms / noise_rms) / (10.0 ** (snr_db / 20.0))
    return (audio + interferer * scale).astype(np.float32)


def convolve_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if rir.size == 0:
        return audio.astype(np.float32, copy=False)
    rir = rir / (np.max(np.abs(rir)) + EPS)
    out = np.convolve(audio, rir, mode="full")[: audio.size]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak * 0.99
    return out.astype(np.float32)


@dataclass
class AugPipeline:
    cfg: AugmentConfig
    sample_rate: int = 16_000
    background_pool: list[Path] | None = None
    rir_pool: list[Path] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)

    def _maybe(self, prob: float) -> bool:
        return bool(self._rng.random() < prob)

    def _rand_range(self, lo: float, hi: float) -> float:
        return float(self._rng.uniform(lo, hi))

    def _load_random(self, pool: list[Path] | None) -> np.ndarray:
        if not pool:
            return np.zeros(0, dtype=np.float32)
        import soundfile as sf  # lazy import; only when pool present
        path = pool[int(self._rng.integers(0, len(pool)))]
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            # cheap linear resample
            ratio = self.sample_rate / sr
            new_len = int(round(wav.size * ratio))
            x = np.linspace(0.0, wav.size - 1, new_len, dtype=np.float64)
            idx0 = np.floor(x).astype(np.int64)
            idx1 = np.clip(idx0 + 1, 0, wav.size - 1)
            frac = (x - idx0).astype(np.float32)
            wav = ((1.0 - frac) * wav[idx0] + frac * wav[idx1]).astype(np.float32)
        return wav

    def apply(self, audio: np.ndarray) -> np.ndarray:
        if not self.cfg.enable:
            return audio.astype(np.float32, copy=False)

        out = audio.astype(np.float32, copy=True)

        if self._maybe(self.cfg.pitch_prob):
            st = self._rand_range(*self.cfg.pitch_semitones_range)
            out = pitch_shift_resample(out, st)

        if self._maybe(self.cfg.time_shift_prob):
            ms = self._rand_range(*self.cfg.time_shift_ms_range)
            out = time_shift(out, int(round(ms * self.sample_rate / 1000.0)))

        if self._maybe(self.cfg.rir_prob):
            rir = self._load_random(self.rir_pool)
            if rir.size > 0:
                out = convolve_rir(out, rir)

        if self._maybe(self.cfg.background_prob):
            bg = self._load_random(self.background_pool)
            if bg.size > 0:
                snr = self._rand_range(*self.cfg.background_snr_db_range)
                out = mix_at_snr(out, bg, snr)

        if self._maybe(self.cfg.noise_prob):
            snr = self._rand_range(*self.cfg.noise_snr_db_range)
            out = add_white_noise(out, snr, self._rng)

        if self._maybe(self.cfg.gain_prob):
            db = self._rand_range(*self.cfg.gain_db_range)
            out = random_gain(out, db)

        return out
