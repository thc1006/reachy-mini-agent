# reachy-mini-agent

**Real-time voice + vision AI agent for the [Reachy Mini](https://github.com/pollen-robotics/reachy-mini) robot.**
Runs fully on your own GPU: local LLM (Ollama / Qwen3), Whisper STT, Kokoro or Microsoft Edge TTS, MediaPipe hand gestures, YuNet face tracking, WebRTC for audio/video — optionally bridged over Tailscale so the brain can live on a beefy server while the robot stays in the living room.

> **Status**: working prototype. Conversation loop, face tracking, gesture recognition, and multimodal scene understanding (vision-language model every 10 s) are all online. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline.

---

## What it does

1. **Sees you.** YuNet ONNX face detector drives head pose (`set_target` @ 50 Hz). MediaPipe hand landmarker reacts to finger counts.
2. **Listens to you.** Robot's USB mic → WebRTC audio → local `faster-whisper` (large-v3-turbo, int8_float16) on GPU.
3. **Thinks.** Local Ollama (Qwen3-8B by default) produces JSON with `{"speech": "...", "actions": [...]}`. Every 10 s, Qwen2.5-VL:7B describes the camera view and the description is injected (sandboxed) into the system prompt, so the robot can naturally reference what it sees.
4. **Speaks.** Microsoft Ana (edge-tts, cloud) or Kokoro `af_heart` (local ONNX). Responses are cached to disk with LRU eviction — repeated phrases replay in ~20 ms.
5. **Moves.** Head tracking + a library of expressive actions (nod, shake, greet, happy, think…).

---

## Quickstart

```bash
git clone https://github.com/thc1006/reachy-mini-agent.git
cd reachy-mini-agent

# 1. Python env (uv or pip — pick one)
uv venv && uv pip install -e . -e ".[servers,kokoro]"

# 2. Fetch perception models (~8 MB)
curl -L -o face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# 3. (Optional) Fetch Kokoro TTS models if you want local voice
curl -L -o kokoro-v1.0.onnx   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# 4. Configure
cp .env.example .env
#   → edit REACHY_HOST, pick TTS_ENGINE, etc.

# 5. Run the three services (in three terminals, or via systemd units in systemd/)
python src/whisper_server.py    # port 8881
python src/kokoro_server.py     # port 8880 (skip if TTS_ENGINE=edge)
scripts/run_robot.sh            # the main brain
```

Hardware, network layout, and systemd setup — see [docs/HARDWARE.md](docs/HARDWARE.md) and [docs/SETUP.md](docs/SETUP.md).

---

## Architecture at a glance

```
┌──────────────────┐          ┌───────────────────────────────┐
│   Reachy Mini    │          │   Brain host (any Linux box   │
│   (Pollen CM4)   │          │   with a CUDA GPU ≥ 8 GB)     │
│                  │          │                               │
│  ┌────────────┐  │          │  ┌─────────────────────────┐  │
│  │  daemon    │◄─┼──WebRTC──┼──┤ src/robot_brain.py      │  │
│  │  1.6.3     │  │  + WS    │  │ ├ YuNet face detect     │  │
│  │            │  │          │  │ ├ MediaPipe hands       │  │
│  │  USB mic   │  │          │  │ ├ State machine         │  │
│  │  USB spkr  │  │          │  │ └ TTS / STT / LLM glue  │  │
│  │  camera    │  │          │  └─────────────────────────┘  │
│  │  motors    │  │          │  ┌─ local services ────────┐  │
│  └────────────┘  │          │  │ Ollama :11434           │  │
└──────────────────┘          │  │   qwen3:8b (LLM)        │  │
                              │  │   qwen2.5vl:7b (VLM)    │  │
                              │  │ whisper_server :8881    │  │
                              │  │ kokoro_server  :8880    │  │
                              │  └─────────────────────────┘  │
                              └───────────────────────────────┘
```

Under the hood:

- **P2P audio/video** via WebRTC (Opus + H264) negotiated through Pollen's Rust signaling server built into the daemon.
- **Remote control** works over any IP reachable by the brain host — LAN, VPN (Tailscale), or localhost.
- **Vision-language prompt injection** is sandboxed: camera descriptions are fenced in an untrusted block so the LLM doesn't obey text seen through the lens.

---

## Configuration

Every runtime knob is an environment variable. Copy `.env.example` → `.env` and edit:

| Var | Default | Purpose |
|---|---|---|
| `REACHY_HOST` | `reachy-mini.local` | Daemon IP / mDNS name |
| `LLM_MODE` | `ollama` | `ollama` / `litellm` / `claude-sdk` / `claude-cli` |
| `OLLAMA_MODEL` | `qwen3:8b` | Any chat model on your Ollama instance |
| `VISION_MODEL` | `qwen2.5vl:7b` | VLM for scene description |
| `VISION_INTERVAL` | `10` | Seconds between frame captures |
| `TTS_ENGINE` | `edge` | `edge` (Microsoft Ana, cloud) / `kokoro` (local GPU) |
| `KOKORO_VOICE` | `af_heart` | `af_heart` / `af_nicole` / `af_sky` / `af_bella` / … |
| `TTS_PEAK` | `0.95` | Peak-normalize each utterance to this level |
| `TTS_GAIN` | `1.0` | Final gain multiplier (>1 will clip) |
| `TTS_CACHE_MAX_MB` | `50` | LRU cache size limit for edge-tts WAVs |

---

## Hardware we tested on

- **Robot**: Reachy Mini (Pollen Robotics, CM4 variant) running daemon 1.6.3
- **Brain**: Linux server with 2× RTX 3090 (48 GB total VRAM), no NVLink, Ubuntu 24.04
- **Network**: home LAN 1 Gbit + Tailscale mesh for remote brain

Also known to run on a laptop with a single RTX 3050 (4 GB) using `tiny` Whisper + smaller LLM. See [docs/HARDWARE.md](docs/HARDWARE.md) for minimums and GPU allocation guidance.

---

## Why this repo exists

Pollen's official [`reachy_mini_conversation_app`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app) is great, but it's tied to OpenAI Realtime API and leaves your audio/video on someone else's servers. This project is:

- **Fully self-hostable.** No OpenAI / Anthropic calls required (though supported as fallbacks).
- **Designed for fast local GPUs.** Sub-second turn latency on a 3090.
- **Transparent.** Every piece of state is a file or env var; no hidden middleware.

---

## Roadmap / known issues

- [ ] USB audio pipeline from daemon to speaker occasionally silent after long uptime — see issue #1
- [ ] Voice cloning of Ana via XTTS-v2 to bring the cute Microsoft voice fully offline
- [ ] Multi-peer WebRTC (daemon currently gives the media stream to one client at a time)
- [ ] Windows / macOS brain-host support (currently Linux only due to GStreamer plugin paths)

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Third-party notices in [NOTICE](NOTICE).

---

## Credits

- **[Pollen Robotics](https://www.pollen-robotics.com/)** for making Reachy Mini and open-sourcing the daemon, SDK, and signaling stack.
- **[Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)** for a delightful, tiny local TTS.
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** for the best open STT latency on a single GPU.
- **[Ollama](https://ollama.com)** + **Qwen** teams for making local LLM inference frictionless.

Contributions, issues, and PRs welcome.
