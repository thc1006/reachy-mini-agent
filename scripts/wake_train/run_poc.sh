#!/usr/bin/env bash
# POC end-to-end run on the 5090 (Tailscale 100.124.198.14).
# Assumes ~/wake_train venv already exists (piper-tts + openwakeword +
# onnxruntime + numpy + soundfile + torch + huggingface_hub installed).
#
# Run on the 5090 directly:
#     bash scripts/wake_train/run_poc.sh
#
# Or from Windows over ssh:
#     ssh 5090 'bash -s' < scripts/wake_train/run_poc.sh
#
# Outputs:
#     data/wake_train/poc/...   training data
#     artifacts/wake/poc/       trained head + ONNX + report
set -euo pipefail

REPO_ROOT="${WAKE_TRAIN_REPO:-$HOME/reachy-mini}"
VENV="${WAKE_TRAIN_VENV:-$HOME/wake_train}"
PY="$VENV/bin/python"
SCOPE="${SCOPE:-poc}"

cd "$REPO_ROOT"
echo "=== wake_train POC ($SCOPE) ==="
echo "repo:   $REPO_ROOT"
echo "venv:   $VENV"
echo "python: $($PY --version)"
echo

export PYTHONPATH="$REPO_ROOT/src"
"$PY" -m wake_train.cli --verbose negatives --scope "$SCOPE"
"$PY" -m wake_train.cli --verbose synth --scope "$SCOPE"
"$PY" -m wake_train.cli --verbose augment --scope "$SCOPE" --n-per-clip 2
"$PY" -m wake_train.cli --verbose manifest --scope "$SCOPE"
"$PY" -m wake_train.cli --verbose train --scope "$SCOPE"
"$PY" -m wake_train.cli --verbose export --scope "$SCOPE"
"$PY" -m wake_train.cli --verbose eval --scope "$SCOPE"

echo
echo "=== POC done — copy ONNX to Pi 4 and rerun _oww_baseline.py to confirm ==="
ls -lh "$REPO_ROOT/artifacts/wake/$SCOPE/"
