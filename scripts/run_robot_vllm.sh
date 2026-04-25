#!/bin/bash
# robot_brain launcher · s1 2× RTX 3090 · vLLM TP=2 unified backend
# Drop-in replacement for run_robot.sh that flips LLM_BACKEND=vllm
# (continuous batching, no vision/dialog contention).
set -eu

cd "$HOME/dev/reachy-agent/robot"
source .venv/bin/activate

# GStreamer rust plugins live under /opt; on x86_64 GST_PLUGIN_PATH is
# additive so libnice (in /usr/lib/.../gstreamer-1.0/) is still found.
# (On aarch64 the same env var was destructive — symlink-into-system was
# required there. See docs/02-gst-plugins-rs.md for context.)
export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu/gstreamer-1.0

# Robot connection (set via env in production; default Tailscale IP)
export REACHY_HOST=${REACHY_HOST:-100.85.191.3}

# === LLM backend selection ===
# When LLM_BACKEND=vllm, every chat / vision / streaming call routes to the
# vLLM endpoint instead of Ollama (continuous batching = no contention).
export LLM_BACKEND=vllm
export VLLM_HOST=http://localhost:8000
export VLLM_MODEL=qwen36-awq

# Keep these as fallback (LLM_MODE=ollama means ollama route is used; the
# dispatch then checks LLM_BACKEND inside the route)
export LLM_MODE=ollama
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3.6:35b-a3b
export OLLAMA_THINK=0

# Vision: use the same backend (vLLM serves the same multimodal model)
export VISION_URL=$VLLM_HOST
export VISION_MODEL=$VLLM_MODEL
export VISION_INTERVAL=10        # vLLM concurrent batching makes 30s overkill; 10s = fresher scene
export LLM_STREAMING=1
export LLM_TOOLS=1

# TTS
export TTS_ENGINE=edge
export TTS_GAIN=0.50
export STT_DUMP=1     # save every captured audio to /tmp/stt_dump/ for inspection

# STT / Kokoro
export WHISPER_URL=http://localhost:8881
export KOKORO_URL=http://localhost:8880

# Mem0
export ROBOT_MEMORY=1
export MEM0_USER_ID=default
export MEM0_LLM_MODEL=qwen3.6:35b-a3b
export MEM0_EMBED_MODEL=bge-m3

# (Optional) Cascaded perception
export VISION_CASCADE=${VISION_CASCADE:-0}        # 1 to enable YOLOv8n background detector
export CASCADE_INTERVAL=${CASCADE_INTERVAL:-0.5}
export CASCADE_VERBOSE=${CASCADE_VERBOSE:-0}

# CPU
export OMP_NUM_THREADS=4

exec python robot_brain.py
