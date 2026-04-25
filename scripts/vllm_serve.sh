#!/bin/bash
# vLLM serve · Qwen3.6-35B-A3B AWQ on s1 2× RTX 3090
# Production launch script. See docs for benchmark methodology.
set -eu

MODEL_PATH=${MODEL_PATH:-/home/reachym/models/qwen36-awq}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}    # leave room for robot_brain Whisper (1.6GB on GPU0)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}    # vision + dialog + 1 prewarm + 1 headroom

# QuantTrio recommended env (from model card)
export VLLM_USE_DEEP_GEMM=0          # Hopper+ only
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4

# Critical: Ampere SM 8.6 has no native FP8 — disable FP8 paths
# (TurboQuant FP8 broken on Ampere per vllm#40124)
export VLLM_DISABLE_FP8=1

source /home/reachym/venvs/vllm/bin/activate

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
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --trust-remote-code \
    --host 127.0.0.1 --port "$PORT"
    # NOTE: --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    # was tested 2026-04-25 on this hardware: mean 111 tok/s (vs no-MTP 126),
    # variance 75-142 tok/s. MTP acceptance rate not high enough for diverse
    # voice prompts to net a win. Revert and document.
