#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

mkdir -p logs/
now=$(date +"%Y%m%d_%H%M%S")

# v2:
# - routes LLMSectionHeaderProcessor to dedicated Qwen 32B model instead of OlmOCR 7B
# - removes form processor from marker
# - auto-detect useless pages at front and back of each PDF.
# - downscale images to 1288 pixels based on OlmOCR docs:
#   -> speeds up LLM calls a lot as image token count is greatly reduced!
python3 -m scripts.process_html_to_markdown \
    --html-dir "./data/sec_filings/raw_htmls/" \
    --output-dir "./data/sec_filings_md_v2/" \
    --year-cutoff 2022 \
    --drop-front-pages -1 \
    --drop-back-pages -1 \
    --openai-model allenai/olmOCR-2-7B-1025-FP8 \
    --token-count-hf-model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --timeout 120 \
    --sectionheader-openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --sectionheader-token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
    --sectionheader-timeout 360 \
    --max-retries 1 \
    --log-prompt-token-count \
    --gpu-ids 0,1 \
    --workers 4 \
    --max-concurrency 16 \
    --openai-temperature 0.1 \
    --max-image-long-side 1288 \
    2>&1 | tee logs/process_v2_$now.log

# v1 pipeline below
# --openai-model allenai/olmOCR-2-7B-1025 \
# --openai-base-url http://0.0.0.0:8989/v1 \
# python3 -m scripts.process_html_to_markdown \
#     --html-dir "./data/sec_filings/raw_htmls/" \
#     --output-dir "./data/sec_filings/" \
#     --year-cutoff 2022 \
#     --openai-model allenai/olmOCR-2-7B-1025-FP8 \
#     --token-count-hf-model-id Qwen/Qwen2.5-VL-7B-Instruct \
#     --log-prompt-token-count \
#     --gpu-ids 0,1 \
#     --workers 4 \
#     --timeout 200 \
#     --max-retries 1 \
#     --max-concurrency 16 \
#     --openai-temperature 0.1 \
#     2>&1 | tee logs/process_$now.log
