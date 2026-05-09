#!/bin/bash
# vLLM serve · Qwen3.6-35B-A3B AWQ on s1 2× RTX 3090
# Production launch script. See docs for benchmark methodology.
#
# Default config = v4.0 winner (2026-05-07): MTP k=3 + prefix-cache OFF.
# Reference: https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v4.0
#
# Env toggles (override on launch — bash MTP_K=2 ./vllm_serve.sh):
#   MTP_K                  default 3 — number of MTP speculative tokens.
#                          Set 0 to disable spec decoding (v1/v2 baseline).
#                          k=1 = v3 canonical, k=2 = pre-v4 production,
#                          k=3 = v4 winner (TTFT −33 % vs k=2, TPOT NS).
#   ENABLE_PREFIX_CACHING  default 0 — cache OFF when MTP enabled.
#                          Set 1 only if MTP_K=0 OR you have a workload
#                          where vllm-project/vllm#38182 (cache hit rate
#                          92 %→71 % with MTP on Qwen3.6 MoE) does not
#                          dominate.

set -eu

MODEL_PATH=${MODEL_PATH:-/home/reachym/models/qwen36-awq}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}    # leave room for robot_brain Whisper (1.6GB on GPU0)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}    # vision + dialog + 1 prewarm + 1 headroom

MTP_K=${MTP_K:-3}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}

# QuantTrio recommended env (from model card)
export VLLM_USE_DEEP_GEMM=0          # Hopper+ only
export VLLM_USE_FLASHINFER_MOE_FP16=1   # NOTE: vllm 0.20.1 raises NotImplementedError if set; unset for 0.20.x
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4

# Critical: Ampere SM 8.6 has no native FP8 — disable FP8 paths
# (TurboQuant FP8 broken on Ampere per vllm#40124)
export VLLM_DISABLE_FP8=1

source /home/reachym/venvs/vllm/bin/activate

# Build optional flags from env toggles
SPEC_FLAGS=()
if [ "${MTP_K}" -gt 0 ]; then
    SPEC_FLAGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
fi

if [ "${ENABLE_PREFIX_CACHING}" -eq 1 ]; then
    CACHE_FLAG=(--enable-prefix-caching)
else
    CACHE_FLAG=(--no-enable-prefix-caching)
fi

# TP=2 across both 3090s. Expert parallel keeps MoE balanced across cards.
# Chunked prefill + priority scheduling: dialog decodes get priority over
# vision prefills — solves the contention problem this whole exercise is for.
exec vllm serve "$MODEL_PATH" \
    --served-model-name qwen36-awq \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-chunked-prefill \
    "${CACHE_FLAG[@]}" \
    "${SPEC_FLAGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --trust-remote-code \
    --host 127.0.0.1 --port "$PORT"
