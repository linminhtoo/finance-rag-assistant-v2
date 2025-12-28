#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

now=$(date +"%Y%m%d_%H%M%S")

# TorchInductor/Triton default to `/tmp`, which can be unwritable on shared systems
# (e.g. stale `/tmp/torchinductor_$USER` owned by someone else). Force caches into
# the user's home dir to avoid PermissionError.
: "${XDG_CACHE_HOME:=$HOME/.cache}"
: "${TORCHINDUCTOR_CACHE_DIR:=$XDG_CACHE_HOME/torchinductor}"
: "${TRITON_CACHE_DIR:=$XDG_CACHE_HOME/triton}"
: "${TMPDIR:=$XDG_CACHE_HOME/tmp}"
export XDG_CACHE_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TMPDIR
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
: "${CUDA_VISIBLE_DEVICES:=0,1}"

# see: https://docs.vllm.ai/en/latest/examples/online_serving/opentelemetry/
: "${OTEL_SERVICE_NAME:=vllm-server}"
: "${OTEL_EXPORTER_OTLP_TRACES_INSECURE:=true}"
export OTEL_SERVICE_NAME OTEL_EXPORTER_OTLP_TRACES_INSECURE
: "${VLLM_API_KEY:=test}"
: "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:=grpc://127.0.0.1:4317}"

vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --mm-encoder-attn-backend TORCH_SDPA \
    --tensor-parallel-size 2 \
    --max-model-len 28000 \
    --gpu-memory-utilization 0.82 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --host 0.0.0.0 \
    --port 8993 \
    --api-key test \
    --otlp-traces-endpoint http://localhost:4318/v1/traces \
    2>&1 | tee serve_vllm_SDPA_qwen3_vl_32b_instruct_fp8_$now.log

# Olmo's finetuned and RL'ed document understanding model, objectively inferior to Qwen3-VL
# vllm serve allenai/olmOCR-2-7B-1025-FP8 ...
