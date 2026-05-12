"""Piper TTS positive synthesis with sampling jitter.

Piper itself is deterministic, so to get phrase variants beyond pure voice
diversity we vary three sampling knobs: length_scale (pace), noise_scale
(timbre noise), and noise_w (phoneme width noise). The product gives prosody
variation without needing model swap.

Voice models are loaded from VOICE_DIR (default ~/wake_train/voices on the
5090). Each (voice, phrase, jitter_seed) triple writes one wav into
out_dir/<voice>/<phrase_slug>/<seed>.wav at the configured sample rate.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..phrases import Phrase

log = logging.getLogger(__name__)


VOICE_DIR = Path(os.environ.get("WAKE_TRAIN_VOICES", str(Path.home() / "wake_train" / "voices")))


CLIP_DURATION_S: float = 2.0  # pad every positive to >= openWakeWord 16-frame
                              # window (16 * 80 ms = 1.28 s); 2.0 s leaves room
                              # for time-shift augmentation and runtime jitter


@dataclass(frozen=True)
class SynthSpec:
    voice: str
    phrase: Phrase
    seed: int
    length_scale: float
    noise_scale: float
    noise_w: float

    def filename(self) -> str:
        return f"{self.seed:04d}.wav"


def _pad_to_duration(pcm16: np.ndarray, sample_rate: int, duration_s: float,
                     rng: np.random.Generator) -> np.ndarray:
    target = int(round(duration_s * sample_rate))
    if pcm16.size >= target:
        # already long enough — center-crop so the phrase stays inside
        start = (pcm16.size - target) // 2
        return pcm16[start:start + target]
    # random leading silence so the phrase doesn't always sit flush-left
    head = int(rng.integers(0, target - pcm16.size + 1))
    out = np.zeros(target, dtype=pcm16.dtype)
    out[head:head + pcm16.size] = pcm16
    return out


def _jitter_params(rng: np.random.Generator) -> tuple[float, float, float]:
    length_scale = float(rng.uniform(0.85, 1.20))
    noise_scale = float(rng.uniform(0.45, 0.80))
    noise_w = float(rng.uniform(0.50, 0.95))
    return length_scale, noise_scale, noise_w


def plan(
    voices: Iterable[str],
    phrases: Iterable[Phrase],
    n_per_voice_phrase: int,
    seed: int = 2026,
) -> list[SynthSpec]:
    rng = np.random.default_rng(seed)
    specs: list[SynthSpec] = []
    for voice in voices:
        for phrase in phrases:
            for i in range(n_per_voice_phrase):
                ls, ns, nw = _jitter_params(rng)
                specs.append(SynthSpec(voice, phrase, i, ls, ns, nw))
    return specs


def _voice_path(voice: str) -> tuple[Path, Path]:
    """Return (model_path, config_path) for a Piper voice name."""
    model = VOICE_DIR / f"{voice}.onnx"
    cfg = VOICE_DIR / f"{voice}.onnx.json"
    if not model.exists() or not cfg.exists():
        raise FileNotFoundError(
            f"Piper voice {voice!r} missing under {VOICE_DIR}; "
            f"expected {model.name} + {model.name}.json"
        )
    return model, cfg


def synthesize_phrase_set(
    voices: tuple[str, ...],
    phrases: tuple[Phrase, ...],
    out_dir: Path,
    n_per_voice_phrase: int,
    sample_rate: int = 16_000,
    seed: int = 2026,
) -> Path:
    """Synthesize every (voice, phrase) x n_per_voice_phrase jittered clip.

    Returns the manifest path. Manifest is a JSON list of records:
        {voice, slug, text, lang, seed, params, wav}
    Lazily imports piper-tts so unit tests don't need it installed.
    """
    from piper import PiperVoice  # lazy
    from piper.config import SynthesisConfig
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    records: list[dict] = []

    voice_cache: dict[str, PiperVoice] = {}
    pad_rng = np.random.default_rng(seed + 7)

    for spec in plan(voices, phrases, n_per_voice_phrase, seed=seed):
        if spec.voice not in voice_cache:
            model_p, cfg_p = _voice_path(spec.voice)
            voice_cache[spec.voice] = PiperVoice.load(str(model_p), config_path=str(cfg_p))
        v = voice_cache[spec.voice]

        wav_dir = out_dir / spec.voice / spec.phrase.slug
        wav_dir.mkdir(parents=True, exist_ok=True)
        wav_path = wav_dir / spec.filename()

        syn_cfg = SynthesisConfig(
            length_scale=spec.length_scale,
            noise_scale=spec.noise_scale,
            noise_w_scale=spec.noise_w,
        )
        parts: list[np.ndarray] = []
        voice_sr: int | None = None
        for chunk in v.synthesize(spec.phrase.text, syn_config=syn_cfg):
            parts.append(np.asarray(chunk.audio_int16_array, dtype=np.int16))
            voice_sr = chunk.sample_rate
        if not parts or voice_sr is None:
            raise RuntimeError(f"Piper produced no audio for {spec.voice}/{spec.phrase.slug}")
        pcm16 = np.concatenate(parts)

        if voice_sr != sample_rate:
            # Anti-aliased polyphase resample (Piper's 22050 -> 16000 is a
            # downsample; naive linear interp folds anything above 8 kHz back
            # into the band and gave the wake-detection model a Piper-only
            # alias signature to latch onto).
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sample_rate, voice_sr)
            resampled = resample_poly(pcm16.astype(np.float32),
                                       sample_rate // g, voice_sr // g)
            np.clip(resampled, -32768, 32767, out=resampled)
            pcm16 = resampled.astype(np.int16)

        pcm16 = _pad_to_duration(pcm16, sample_rate, CLIP_DURATION_S, pad_rng)
        sf.write(str(wav_path), pcm16, sample_rate, subtype="PCM_16")
        records.append({
            "voice": spec.voice,
            "slug": spec.phrase.slug,
            "text": spec.phrase.text,
            "lang": spec.phrase.lang,
            "seed": spec.seed,
            "params": {
                "length_scale": spec.length_scale,
                "noise_scale": spec.noise_scale,
                "noise_w": spec.noise_w,
            },
            "wav": str(wav_path.relative_to(out_dir)),
        })

    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    log.info("synth complete: %d clips -> %s", len(records), out_dir)
    return manifest_path
