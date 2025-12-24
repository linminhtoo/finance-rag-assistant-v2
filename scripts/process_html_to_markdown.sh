#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

mkdir -p logs/

# --openai-model allenai/olmOCR-2-7B-1025 \
# --openai-base-url http://0.0.0.0:8989/v1 \
now=$(date +"%Y%m%d_%H%M%S")
python3 -m scripts.process_html_to_markdown \
    --html-dir "./data/sec_filings/raw_htmls/" \
    --output-dir "./data/sec_filings/" \
    --year-cutoff 2022 \
    --openai-model allenai/olmOCR-2-7B-1025-FP8 \
    --hf-model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --log-prompt-token-count \
    --gpu-ids 0,1 \
    --workers 4 \
    --timeout 200 \
    --max-retries 1 \
    --max-concurrency 16 \
    --openai-temperature 0.1 \
    2>&1 | tee logs/process_$now.log
