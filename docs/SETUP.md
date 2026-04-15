# Setup

End-to-end walkthrough: clone → services up → talking robot. Assumes a Linux brain host with a CUDA GPU, Ollama installed, Reachy Mini running firmware ≥ 1.6.3, and `gst-plugins-rs` built per [HARDWARE.md](HARDWARE.md).

## 1. Clone + install

```bash
git clone https://github.com/thc1006/reachy-mini-agent.git
cd reachy-mini-agent

# uv is faster, but pip works too
uv venv --python 3.12
uv pip install -e ".[servers,kokoro]"
```

## 2. Fetch models

```bash
# Vision / hand models (~8 MB)
curl -L -o face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Kokoro TTS (~350 MB, skip if TTS_ENGINE=edge)
curl -L -o kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Whisper model is auto-downloaded on first run (~1.6 GB to ~/.cache/huggingface)

# Ollama models (~12 GB)
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
```

## 3. Configure

```bash
cp .env.example .env
$EDITOR .env
```

Key values to set:

- `REACHY_HOST` — your robot's IP / mDNS name (try `reachy-mini.local` first)
- `TTS_ENGINE` — `edge` for the cute Microsoft Ana voice (needs internet), `kokoro` for local
- `OLLAMA_MODEL` / `VISION_MODEL` — any models you've pulled

## 4. Smoke-test services individually

```bash
# In three terminals:
.venv/bin/python src/whisper_server.py
.venv/bin/python src/kokoro_server.py    # only if TTS_ENGINE=kokoro
scripts/run_robot.sh
```

You should hear: _"System online! I'll say hi to anyone who comes by!"_ from the robot within ~20 seconds. Walk in front of the camera — the head should track and a greeting should play.

## 5. Pre-warm the TTS cache (optional but recommended)

```bash
.venv/bin/python src/prewarm_tts_cache.py
```

This fetches edge-tts WAVs for the 34 static phrases + the top-N most frequent past utterances from `conversation_log.jsonl`. Future plays of those phrases hit disk instead of the cloud (~20 ms vs ~300 ms).

## 6. systemd (optional but recommended for permanence)

```bash
mkdir -p ~/.config/systemd/user ~/reachy-mini-agent/logs
cp systemd/*.example ~/.config/systemd/user/
rename 's/\.example$//' ~/.config/systemd/user/*.example   # or cp each individually

# Enable user-level systemd to survive logout
sudo loginctl enable-linger "$USER"

systemctl --user daemon-reload
systemctl --user enable --now kokoro whisper robot-brain
```

Logs appear in `~/reachy-mini-agent/logs/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gst_plugin_webrtc_signalling` build error | You need cuDNN 9. See [HARDWARE.md](HARDWARE.md#cuda--cudnn). |
| `webrtcsrc` element not found | `GST_PLUGIN_PATH` not set or `gst-plugins-rs` not installed. |
| `Failed to connect to ws://<LAN-IP>:8000/ws/sdk: timed out` | `REACHY_HOST` points to an IP that's not reachable from the brain. Pick LAN IP if on same Wi-Fi, Tailscale IP if remote. |
| Head hangs limp | Motors not woken. `curl -X POST http://$REACHY_HOST:8000/api/motors/set_mode/enabled && curl -X POST http://$REACHY_HOST:8000/api/move/play/wake_up`. |
| No sound from robot even though logs say TTS sent | Daemon's GStreamer pipeline occasionally gets wedged after many peer churns. `sudo systemctl restart reachy-mini-daemon`. |
| `NoAudioReceived` from edge-tts | Microsoft Azure cloud reject; check the phrase isn't just punctuation/emoji. Falls back to Kokoro automatically. |
