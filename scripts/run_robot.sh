#!/bin/bash
cd /home/reachym/dev/reachy-agent/robot
SP=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib
export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu
export REACHY_HOST=100.85.191.3
export LLM_MODE=ollama
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3.6:35b-a3b
export OLLAMA_THINK=0
export VISION_MODEL=qwen3.6:35b-a3b   # unify: same MoE serves both dialog and VL (saves ~6 GB)
export LLM_STREAMING=1                # sentence-by-sentence streaming → faster first-audio
export LLM_TOOLS=1                    # enable Ollama tool-calling in _ask_via_ollama
export TTS_ENGINE=edge
export TTS_GAIN=0.35
export KOKORO_URL=http://localhost:8880
export WHISPER_URL=http://localhost:8881
exec .venv/bin/python robot_brain.py
