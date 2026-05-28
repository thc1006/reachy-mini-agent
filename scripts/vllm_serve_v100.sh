#!/usr/bin/env bash
# vllm0528 (TWCC NGC container, 2× V100-SXM2-32GB) production launch.
#
# This script encodes the empirically-validated 2026-05-28 working config:
#   - --enforce-eager   : CUDA Graphs disabled. FULL graph capture crashes on
#                          1Cat-vLLM v1.0.0 × MTP × hybrid Linear+Full attention
#                          (cudaErrorStreamCaptureInvalidated at FULL capture).
#   - NO --speculative-config : MTP retried 2026-05-28, k=2/3/4 acceptance ~0%,
#                                speedup 1.2× = within noise. Don't enable.
#   - --gpu-memory-utilization 0.45 : two zombie tenant PIDs hold 16GB/card on
#                                      TWCC; practical free VRAM ≈ 16GB → 0.45 fits.
#   - VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 : Volta SM 7.0 backend from 1Cat fork.
#
# Operators: do NOT add MTP_K=... / --speculative-config on V100. See
# memory project_vllm_38182_contribution_2026_05_17 and the 2026-05-28
# MTP-RETRY empirical finding.
#
# Counterpart for 2× RTX 3090 (s1, currently dead): scripts/vllm_serve.sh.

set -eu

MODEL_PATH=${MODEL_PATH:-/home/hctsai1006/models/Qwen3.6-35B-A3B-AWQ}
SERVED_NAME=${SERVED_NAME:-qwen36-awq}
VENV=${VENV:-/home/hctsai1006/venvs/v100-vllm}
LOG_DIR=${LOG_DIR:-$HOME/vllm-logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/serve_v100.log}

TP=${TP:-2}
GPU_UTIL=${GPU_UTIL:-0.45}
MAX_LEN=${MAX_LEN:-8192}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

mkdir -p "$LOG_DIR"

# Hard-required for Volta SM 7.0 paths in 1Cat-vLLM v1.0.0.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100

# VRAM-cleanup guard: any leftover api_server from a previous launch must die
# before relaunch, else CUDA OOM on TP=2 init. See feedback_vram_cleanup_before_bench.
if pgrep -f 'vllm.entrypoints.openai.api_server' >/dev/null; then
  echo "[vllm_serve_v100] killing previous api_server pids:"
  pgrep -f 'vllm.entrypoints.openai.api_server' || true
  pkill -9 -f 'vllm.entrypoints.openai.api_server' || true
  sleep 5
fi

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host "$HOST" \
  --port "$PORT"
