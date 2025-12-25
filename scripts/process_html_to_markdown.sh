#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
export LOGLEVEL=DEBUG

mkdir -p logs/
now=$(date +"%Y%m%d_%H%M%S")

# v4:
# - add option to control depth of LLM analysis for SectionHeader & Table Processors.
# - use Qwen3 VL for every task, especially PageCorrection might be brittle.
# NOTE: interrupted due to vLLM server OOM.
# v4b:
# - skip Table Merge processor. it is unnecessary and just wastes LLM calls.
python3 -m scripts.process_html_to_markdown \
    --html-dir "./data/sec_filings/raw_htmls/" \
    --output-dir "./data/sec_filings_md_v4b/" \
    --year-cutoff 2022 \
    --drop-front-pages -1 \
    --drop-back-pages -1 \
    --openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
    --timeout 240 \
    --sectionheader-openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --sectionheader-token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
    --sectionheader-timeout 500 \
    --max-retries 1 \
    --log-prompt-token-count \
    --gpu-ids 0,1 \
    --workers 2 \
    --max-concurrency 16 \
    --openai-temperature 0 \
    --max-image-long-side 1288 \
    --image-expansion-ratio 0.05 \
    --analysis-style auto \
    --disable-forms \
    --disable-table-merge \
    2>&1 | tee logs/process_v4b_$now.log

# python3 -m scripts.process_html_to_markdown \
#     --html-dir "./data/htmls_for_debugging/" \
#     --output-dir "./data/sec_filings_md_v4b_debug/" \
#     --year-cutoff 2022 \
#     --drop-front-pages -1 \
#     --drop-back-pages -1 \
#     --openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
#     --token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
#     --timeout 240 \
#     --sectionheader-openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
#     --sectionheader-token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
#     --sectionheader-timeout 500 \
#     --max-retries 1 \
#     --log-prompt-token-count \
#     --gpu-ids 0,1 \
#     --workers 2 \
#     --max-concurrency 16 \
#     --openai-temperature 0 \
#     --max-image-long-side 1288 \
#     --image-expansion-ratio 0.05 \
#     --analysis-style auto \
#     --disable-forms \
#     --disable-table-merge \
#     2>&1 | tee logs/process_v4b_$now.log

# v3:
# - major: add more robust schema, prompt and parsing of structured JSON output.
# - major: fixed bug in LLMPageCorrectionProcessor, allowing it to actually run (it was skipping cuz user_prompt was None)
# - major: fix bbox expansion mismatch bug in LLMTableProcessor when image_expansion_ratio is used
# - try --image-expansion-ratio 0.05
# - remove TOC artifacts from the top of each PDF page before sending to marker
# - match marker's preferred font settings for weasyprint HTML -> PDF step

# DOING: test on a few debug HTMLs first and check the error cases are gone
# python3 -m scripts.process_html_to_markdown \
#     --html-dir "./data/htmls_for_debugging/" \
#     --output-dir "./data/sec_filings_md_v3_debug/" \
#     --year-cutoff 2022 \
#     --drop-front-pages -1 \
#     --drop-back-pages -1 \
#     --openai-model allenai/olmOCR-2-7B-1025-FP8 \
#     --token-count-hf-model-id Qwen/Qwen2.5-VL-7B-Instruct \
#     --timeout 120 \
#     --sectionheader-openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
#     --sectionheader-token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
#     --sectionheader-timeout 360 \
#     --max-retries 1 \
#     --log-prompt-token-count \
#     --gpu-ids 0,1 \
#     --workers 4 \
#     --max-concurrency 16 \
#     --openai-temperature 0.1 \
#     --max-image-long-side 1288 \
#     --image-expansion-ratio 0.05 \
#     --analysis-style auto \
#     2>&1 | tee logs/process_v3_$now.log

# interesting, some prompts in TableSchema do not have any image tokens ...? why ?
# 2025-12-25 19:44:18.984 | INFO     | scripts.process_html_to_markdown:__call__:163 - CustomOpenAIService[20f23f1aa50c42a98dec60805c6abb17] start timeout=120s max_retries=1 model=allenai/olmOCR-2-7B-1025-FP8 schema=marker.processors.llm.llm_table.TableSchema
# 2025-12-25 19:44:59,941 [INFO] marker: CustomOpenAIService[20f23f1aa50c42a98dec60805c6abb17] success elapsed_s=40.957 attempt=1/2 prompt_tokens=1823 completion_tokens=85 total_tokens=1908
# 2025-12-25 19:44:59,944 [INFO] marker: CustomOpenAIService[d20e2725126e4b7c90b9d3e3059e5c81] success elapsed_s=40.974 attempt=1/2 prompt_tokens=1134 completion_tokens=85 total_tokens=1219
# 2025-12-25 19:44:59,947 [INFO] marker: CustomOpenAIService[193f9ae80b9347598573fa53956244b1] success elapsed_s=41.003 attempt=1/2 prompt_tokens=2138 completion_tokens=85 total_tokens=2223


# v2:
# - routes LLMSectionHeaderProcessor to dedicated Qwen 32B model instead of OlmOCR 7B
# - removes form processor from marker
# - auto-detect useless pages at front and back of each PDF.
# - downscale images to 1288 pixels based on OlmOCR docs:
#   -> speeds up LLM calls a lot as image token count is greatly reduced!
# PROBLEMS:
# - images of tables are cropped on the right edge. need to add padding/margin.
#   -> try to fix by setting --image-expansion-ratio 0.05
# python3 -m scripts.process_html_to_markdown \
#     --html-dir "./data/sec_filings/raw_htmls/" \
#     --output-dir "./data/sec_filings_md_v2/" \
#     --year-cutoff 2022 \
#     --drop-front-pages -1 \
#     --drop-back-pages -1 \
#     --openai-model allenai/olmOCR-2-7B-1025-FP8 \
#     --token-count-hf-model-id Qwen/Qwen2.5-VL-7B-Instruct \
#     --timeout 120 \
#     --sectionheader-openai-model Qwen/Qwen3-VL-32B-Instruct-FP8 \
#     --sectionheader-token-count-hf-model-id Qwen/Qwen3-VL-32B-Instruct \
#     --sectionheader-timeout 360 \
#     --max-retries 1 \
#     --log-prompt-token-count \
#     --gpu-ids 0,1 \
#     --workers 4 \
#     --max-concurrency 16 \
#     --openai-temperature 0.1 \
#     --max-image-long-side 1288 \
#     2>&1 | tee logs/process_v2_$now.log

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
