#!/bin/bash
# robot_brain launcher · DGX Spark (GB10) · LOW VOLUME profile
set -eu

cd "$HOME/dev/reachy-agent"

# Activate aarch64 robot venv
source .venv/bin/activate

# GStreamer rust plugins are now symlinked into /usr/lib/.../gstreamer-1.0,
# so no GST_PLUGIN_PATH needed (and setting it overrides the system path,
# which hides libnice elements that webrtcbin requires for ICE).

# CUDA 13 toolchain on path so libcudart etc. resolve
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Robot connection
export REACHY_HOST=100.85.191.3   # Tailscale

# LLM — Ollama daemon already running with auto-loaded qwen3.6:35b-a3b
export LLM_MODE=ollama
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3.6:35b-a3b
export OLLAMA_THINK=0
export VISION_MODEL=qwen3.6:35b-a3b      # unified VL/dialog (saves ~15 GB)
export LLM_STREAMING=1
export LLM_TOOLS=1

# TTS — edge-tts (cloud) is fastest; Kokoro CPU as fallback
export TTS_ENGINE=edge
# Louder profile (revised live during test)
export TTS_GAIN=0.60
# WebRTC + GStreamer + USB audio still needs the 1.2s tail pad
# (set inside robot_brain.py constants — keep at 1.2)

# STT / TTS local services (whisper auto-loads, kokoro optional)
export WHISPER_URL=http://localhost:8881
export KOKORO_URL=http://localhost:8880

# Mem0 long-term memory
export ROBOT_MEMORY=1
export MEM0_USER_ID=default
export MEM0_LLM_MODEL=qwen3.6:35b-a3b
export MEM0_EMBED_MODEL=bge-m3

# Threading caps for the GB10 ARM CPU
export OMP_NUM_THREADS=8

exec python src/robot_brain.py
