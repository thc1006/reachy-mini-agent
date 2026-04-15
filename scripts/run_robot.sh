#!/usr/bin/env bash
# Reachy Mini Agent launcher — source .env if present, resolve CUDA libs, run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
VENV="${VENV:-.venv}"

# Load .env if it exists (don't fail if missing)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# CUDA runtime libs shipped in the wheel (for CTranslate2 / onnxruntime-gpu)
SP="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

# GStreamer Rust plugins (built manually from pollen-robotics fork on host)
export GST_PLUGIN_PATH="${GST_PLUGIN_PATH:-/opt/gst-plugins-rs/lib/x86_64-linux-gnu}"

exec "$VENV/bin/python" src/robot_brain.py
