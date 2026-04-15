"""Kokoro TTS FastAPI server — OpenAI-compat /v1/audio/speech on port 8880"""
import io, os, sys, time
from pathlib import Path
import sysconfig

SP = Path(sysconfig.get_paths()["purelib"])
extra_paths = [str(SP / "nvidia" / sub / "lib") for sub in ("cudnn", "cublas", "cuda_nvrtc")]
os.environ["LD_LIBRARY_PATH"] = ":".join(extra_paths + [os.environ.get("LD_LIBRARY_PATH", "")])

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent
MODEL_PATH = str(HERE / "kokoro-v1.0.onnx")
VOICES_PATH = str(HERE / "voices-v1.0.bin")

print("Loading Kokoro on GPU...")
t0 = time.time()
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
print(f"Loaded in {time.time()-t0:.1f}s. Voices: {len(kokoro.get_voices())}")

app = FastAPI(title="Kokoro TTS")

class SpeechReq(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: str = "wav"
    speed: float = 1.0

@app.get("/v1/audio/voices")
async def list_voices():
    return {"voices": list(kokoro.get_voices())}

@app.get("/health")
async def health():
    return {"ok": True, "voices": len(kokoro.get_voices())}

@app.post("/v1/audio/speech")
async def speech(req: SpeechReq):
    try:
        t0 = time.time()
        samples, sr = kokoro.create(req.input, voice=req.voice, speed=req.speed, lang="en-us")
        t_gen = time.time() - t0
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        buf.seek(0)
        wav = buf.read()
        print(f"[TTS] '{req.input[:50]}...' voice={req.voice} gen={t_gen*1000:.0f}ms size={len(wav)}B")
        return Response(content=wav, media_type="audio/wav",
                        headers={"X-Generation-Ms": f"{t_gen*1000:.0f}"})
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880, log_level="warning", access_log=False)
