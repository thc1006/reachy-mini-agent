"""CosyVoice 2 FastAPI wrapper for Reachy Mini elder-care TTS.

Self-host serving CosyVoice2-0.5B on vllm0528 (V100 sm_70, card 0).
Exposes an OpenAI-compatible /v1/audio/speech endpoint so the Pi brain
can route through the same shape it already uses for Kokoro.

Endpoints:
  GET  /health                 — liveness + model info
  POST /v1/audio/speech        — OpenAI-style:
        {"input": str, "voice": str | None,
         "speed": float | None, "language": str | None}
       Returns audio/wav (16-bit PCM, 24 kHz mono).

Threading: CosyVoice2 inference is not thread-safe (per upstream FastAPI
example). We serialize via asyncio.Lock so concurrent HTTP calls don't
collide on the underlying generator/onnxruntime sessions.

Pin to card 0 BEFORE torch initializes (CUDA_VISIBLE_DEVICES).
"""

from __future__ import annotations

import io
import os
import sys
import asyncio
import logging
import time
import wave
from contextlib import asynccontextmanager

# Pin BEFORE torch lazy-init. vlm_server.py uses the same pattern.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

COSYVOICE_REPO = os.environ.get(
    "COSYVOICE_REPO", "/home/hctsai1006/scripts/CosyVoice"
)
COSYVOICE_MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    "/home/hctsai1006/models/cosyvoice/CosyVoice2-0.5B",
)

# CosyVoice repo isn't pip-installed; bolt onto sys.path the same way
# the upstream runtime/python/fastapi/server.py does.
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, "third_party", "Matcha-TTS"))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cosyvoice_server")

# Default zero-shot prompt: the Chinese-female prompt bundled with the
# CosyVoice repo. Soothing mid-range pitch — closest match in-repo to
# what elder-care users have responded well to in HsiaoYu/af_heart bench.
DEFAULT_PROMPT_WAV = os.path.join(COSYVOICE_REPO, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading CosyVoice2 from %s on card 0 (fp16)", COSYVOICE_MODEL_DIR)
    t0 = time.perf_counter()
    app.state.cosy = CosyVoice2(
        COSYVOICE_MODEL_DIR,
        load_jit=False,
        load_trt=False,
        fp16=True,
    )
    app.state.sample_rate = int(app.state.cosy.sample_rate)
    # Per CosyVoice2 frontend (frontend.py lines 96/109/121), prompt_wav is
    # reloaded internally via load_wav() — so it MUST be a path (or file-like
    # opened lazily), not a pre-loaded tensor. The upstream FastAPI sample
    # passes a tensor and is broken for CosyVoice2; we pass the path.
    app.state.prompt_wav = DEFAULT_PROMPT_WAV
    app.state.prompt_text = DEFAULT_PROMPT_TEXT
    app.state.lock = asyncio.Lock()
    app.state.ready = True
    log.info(
        "ready in %.1fs sr=%d prompt_text=%r",
        time.perf_counter() - t0,
        app.state.sample_rate,
        app.state.prompt_text,
    )
    try:
        free, total = torch.cuda.mem_get_info(0)
        log.info(
            "cuda:0 mem free=%.0fMB total=%.0fMB",
            free / 1048576,
            total / 1048576,
        )
    except Exception:
        pass
    yield
    log.info("shutting down")


app = FastAPI(lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = "default"
    speed: float | None = 1.0
    language: str | None = None  # "zh"|"en"|"ja"|"ko"|"auto"; informational only
    model: str | None = None  # OpenAI shape compat — ignored
    response_format: str | None = None  # ignored; we always return wav


def _pcm_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Wrap mono float32 [-1,1] samples in a RIFF/WAV (16-bit PCM) container."""
    # Clip + convert to int16
    audio = np.clip(samples, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@app.get("/health")
async def health():
    ready = bool(getattr(app.state, "ready", False))
    info = {
        "ready": ready,
        "model": "CosyVoice2-0.5B",
        "sample_rate": getattr(app.state, "sample_rate", None),
        "default_prompt_text": getattr(app.state, "prompt_text", None),
    }
    try:
        free, total = torch.cuda.mem_get_info(0)
        info["cuda_free_mb"] = round(free / 1048576, 1)
        info["cuda_total_mb"] = round(total / 1048576, 1)
    except Exception:
        pass
    return info


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    # H2 (CosyVoice review): the readiness check must run BEFORE we touch
    # app.state.lock. Lifespan sets `ready=True` LAST (after CosyVoice2 ctor
    # + lock allocation), so during cold start a request that hits the
    # handler before lifespan completes could grab the lock and stall the
    # event loop while the model is still loading. Hoist the 503 check to
    # the very top of the handler so callers fail fast and the breaker on
    # the brain side trips cleanly during a vllm0528 supervisord restart.
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model not ready")

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")

    speed = float(req.speed or 1.0)
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="speed must be in [0.5, 2.0]")

    cosy = app.state.cosy
    prompt_wav = app.state.prompt_wav
    prompt_text = app.state.prompt_text
    sr = app.state.sample_rate
    lock: asyncio.Lock = app.state.lock

    # M2 (CosyVoice review): language routing is owned by the BRAIN side
    # (_detect_tts_language in src/robot_brain.py classifies via Unicode
    # script counting then forwards "zh"|"en"|"ja"|"ko"). The server only
    # needs to trust that hint — the prior `_is_mostly_non_chinese` heuristic
    # was dead code because the brain never sends "auto", and keeping two
    # diverging classifiers (different thresholds: brain >0 han chars = zh,
    # server <30% han = non-chinese) made the routing impossible to reason
    # about. Trust the hint; default to zero_shot when the hint is missing
    # or unknown — the cn-prompt zero-shot path is the safest default.
    lang_hint = (req.language or "").lower().strip()
    use_cross_lingual = lang_hint in ("en", "ja", "ko")

    t0 = time.perf_counter()
    try:
        wav_bytes, dur_s = await _serialized_inference(
            lock, cosy, text, prompt_text, prompt_wav, speed,
            use_cross_lingual, sr,
        )
    except torch.cuda.OutOfMemoryError as e:
        # M1 (CosyVoice review): split CUDA OOM from generic inference
        # failure so the brain breaker + dashboards can split GPU-pressure
        # events (transient, retriable after backoff) from genuine bugs.
        # empty_cache() releases cached blocks back to the CUDA allocator —
        # the next request may then succeed without a restart.
        log.error("cuda_oom on synth chars=%d: %s", len(text), e)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise HTTPException(
            status_code=503,
            detail="cuda oom",
            headers={"X-Error-Class": "cuda_oom"},
        )
    except HTTPException:
        # Re-raise our own 4xx/5xx (e.g. "no audio produced") untouched —
        # they already carry the right shape; only generic exceptions get
        # the inference_failed wrapper below.
        raise
    except Exception as e:
        log.exception("inference failed chars=%d: %s", len(text), e)
        raise HTTPException(
            status_code=500,
            detail="inference failed",
            headers={"X-Error-Class": "inference_failed"},
        )

    elapsed = time.perf_counter() - t0
    log.info(
        "synth chars=%d audio=%.2fs gen=%.2fs rtf=%.3f mode=%s",
        len(text),
        dur_s,
        elapsed,
        elapsed / max(dur_s, 1e-3),
        "cross_lingual" if use_cross_lingual else "zero_shot",
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration-S": f"{dur_s:.3f}",
            "X-Gen-Latency-S": f"{elapsed:.3f}",
            "X-Mode": "cross_lingual" if use_cross_lingual else "zero_shot",
        },
    )


async def _serialized_inference(
    lock: asyncio.Lock,
    cosy,
    text: str,
    prompt_text: str,
    prompt_wav,
    speed: float,
    cross_lingual: bool,
    sr: int,
) -> tuple[bytes, float]:
    """Hold the per-process lock, run blocking inference in a worker thread,
    and return ``(wav_bytes, duration_seconds)``.

    Extracted from the handler so the invariant ("only one inference at a
    time; the lock wraps both the CosyVoice2 generator AND the onnxruntime
    sessions which are not thread-safe per upstream") lives in one place.
    """
    async with lock:
        chunks = await asyncio.to_thread(
            _run_inference, cosy, text, prompt_text, prompt_wav, speed, cross_lingual
        )
    if not chunks:
        raise HTTPException(status_code=500, detail="no audio produced")
    full = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return _pcm_to_wav(full, sr), len(full) / sr


def _run_inference(cosy, text, prompt_text, prompt_wav, speed, cross_lingual):
    """Blocking helper — drains the CosyVoice2 generator into a list of arrays."""
    out = []
    if cross_lingual:
        gen = cosy.inference_cross_lingual(
            text, prompt_wav, stream=False, speed=speed
        )
    else:
        gen = cosy.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=False, speed=speed
        )
    for chunk in gen:
        speech_t = chunk["tts_speech"]
        # tts_speech is [1, T] torch float on cpu per upstream
        arr = speech_t.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[0]
        out.append(arr.astype(np.float32))
    return out
