#!/bin/bash
cd /home/reachym/dev/reachy-agent/robot
SP=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib
export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu
export REACHY_HOST=100.85.191.3
export LLM_MODE=ollama
# vLLM backend (production since 2026-04-25). Ollama remains the code default
# inside robot_brain.py for safety, but production explicitly routes to vLLM
# here — vLLM has the tool-call parser (qwen3_xml), MTP k=3 spec-decoding, and
# AWQ-Marlin quant that the hardware was benchmarked for. Set LLM_BACKEND=ollama
# to fall back (ollama qwen3.6:35b-a3b must be loaded — won't fit alongside vLLM).
export LLM_BACKEND=vllm
export VLLM_HOST=http://localhost:8000
export VLLM_MODEL=qwen36-awq
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3.6:35b-a3b
export OLLAMA_THINK=0
export VISION_MODEL=qwen3.6:35b-a3b   # unify: same MoE serves both dialog and VL (saves ~6 GB)
export LLM_STREAMING=1                # sentence-by-sentence streaming → faster first-audio
export LLM_TOOLS=1                    # enable tool-calling (works for both vLLM + Ollama backends)
export TTS_ENGINE=edge
export TTS_GAIN=0.35
# Lock to "Google 小姐" zh-TW female voice for all output (including English
# fragments — Chinese-accented English is preferred over voice switching
# mid-conversation). Unset (or set empty) to fall back to pick_voice() auto-routing.
export TTS_VOICE=zh-TW-HsiaoYuNeural
export KOKORO_URL=http://localhost:8880
export WHISPER_URL=http://localhost:8881

# Motor safety (per-frame delta clamp for face tracker; see robot_brain.py
# _clamp_pose_delta). 1.5° at 30 FPS = 45°/sec max angular speed.
export FACE_TRACK_MAX_DELTA_DEG=1.5

# Observability: motor command JSONL log (no-op if path unset).
# Cap at ~100 MB via separate rotation policy (logrotate or similar) —
# at production rates the file grows ~480 MB/day untended.
export MOTOR_LOG_PATH=/home/reachym/dev/reachy-agent/robot/logs/motor.log

exec .venv/bin/python robot_brain.py
