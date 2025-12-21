#!/bin/bash

now=$(date +%Y%m%d_%H%M%S)
python3 scripts/download.py \
    --tickers "APH" "GOOGL" "NVDA" "AMD" "CRDO" "ALAB" \
    --output-dir ./data/sec_filings/ \
    --per-company 10 \
    2>&1 | tee logs/download_${now}.log
