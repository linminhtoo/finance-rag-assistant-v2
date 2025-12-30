#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

now=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs/
python3 -m scripts.run_eval \
  --eval-queries ./eval/eval_queries.jsonl \
  --out-dir ./eval/results/${now} \
  --index-dir ./data/sec_filings_md_v5/chunked_1024_128 \
  --mode normal \
  --concurrency 8 \
  2>&1 | tee logs/run_eval_${now}.log
