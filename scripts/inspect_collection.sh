#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
python3 -m scripts.inspect_collection \
    --backend "milvus" \
    --collection-name with_llm_context_27_12_25 \
    --milvus-uri "/home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128/milvus.db" \
    --max-chars 0 \
    --json \
    --limit 10 \
    2>&1 | tee "logs/inspect_collection_${now}.log"

