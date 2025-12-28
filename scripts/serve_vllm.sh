#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

now=$(date +"%Y%m%d_%H%M%S")
script_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
project_root="$(
  cd -- "$script_dir/.." >/dev/null 2>&1
  pwd
)"

mkdir -p "$project_root/logs"

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
: "${OTEL_EXPORTER_OTLP_TRACES_PROTOCOL:=http/protobuf}"
: "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:=http://localhost:4318/v1/traces}"

vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --mm-encoder-attn-backend TORCH_SDPA \
    --tensor-parallel-size 2 \
    --max-model-len 28000 \
    --gpu-memory-utilization 0.82 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --host 0.0.0.0 \
    --port 8993 \
    --api-key "$VLLM_API_KEY" \
    --otlp-traces-endpoint "$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" \
    2>&1 | tee "$project_root/logs/serve_vllm_SDPA_qwen3_vl_32b_instruct_fp8_$now.log"

# Olmo's finetuned and RL'ed document understanding model, objectively inferior to Qwen3-VL
# vllm serve allenai/olmOCR-2-7B-1025-FP8 ...
