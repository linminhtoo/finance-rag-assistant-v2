#!/bin/bash


mkdir -p logs/
now=$(date +"%Y%m%d_%H%M%S")
python3 scripts/make_eval_set.py \
  --ingest-output-dir ./data/sec_filings_md_v5/chunked_1024_128 \
  --out ./eval/eval_queries.jsonl \
  --max-docs 200 \
  --n-factual 30 \
  --n-open-ended 60 \
  --n-refusal 40 \
  --n-distractor 40 \
  --n-comparison 30 \
  --seed 1337 \
  2>&1 | tee logs/make_eval_set_$now.log
