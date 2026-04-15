"""Whisper STT FastAPI server on 5090 — GPU faster-whisper
POST /transcribe: raw WAV body → {"text": ...}
Port 8881
"""
import io, os, time
from pathlib import Path

# 讓 onnxruntime-like cudnn/cublas 也能被 ctranslate2 找到
import sysconfig
SP = Path(sysconfig.get_paths()["purelib"])
for sub in ("cudnn", "cublas", "cuda_nvrtc"):
    lib = SP / "nvidia" / sub / "lib"
    if lib.exists():
        os.environ["LD_LIBRARY_PATH"] = str(lib) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
COMPUTE    = os.getenv("WHISPER_COMPUTE", "int8_float16")

print(f"Loading faster-whisper {MODEL_NAME} / CUDA / {COMPUTE} ...")
t0 = time.time()
model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE)
print(f"Loaded in {time.time()-t0:.1f}s")

# 預熱
print("Warmup...")
t0 = time.time()
list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
print(f"Warmup {time.time()-t0:.2f}s")

app = FastAPI(title="Whisper STT")

@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_NAME, "compute": COMPUTE}

@app.post("/transcribe")
async def transcribe(req: Request):
    body = await req.body()
    if not body:
        raise HTTPException(400, "empty body")
    try:
        t0 = time.perf_counter()
        # 解 WAV → float32 mono
        data, sr = sf.read(io.BytesIO(body), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            # resample 粗略（scipy 的 signal.resample）
            import scipy.signal
            data = scipy.signal.resample(data, int(len(data) * 16000 / sr)).astype(np.float32)
        audio_s = len(data) / 16000
        segs, _ = model.transcribe(
            data, language="en", beam_size=3, vad_filter=True,
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segs).strip()
        dur = (time.perf_counter() - t0) * 1000
        print(f"[STT] {dur:.0f}ms / {audio_s:.1f}s audio (RTF={dur/(audio_s*1000):.2f}) → '{text[:60]}'")
        return {"text": text, "duration_ms": int(dur), "audio_s": audio_s}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8881, log_level="warning", access_log=False)
