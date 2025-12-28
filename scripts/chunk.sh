#!/bin/bash
set -euo pipefail

# NOTE: script does logging set up internally. we do NOT need to `tee`
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# 4.5 mins for processed_files=119 total_chunks=15562
python3 -m scripts.chunk \
	    --markdown-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/processed_markdown \
	    --output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128 \
	    --preprocess-markdown-tables \
		--max-tokens 1024 \
		--overlap-tokens 128 \
	    --recursive

# python3 -m scripts.chunk \
# 	    --markdown-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings/processed_markdown \
# 	    --output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings/chunked_v1 \
# 	    --preprocess-markdown-tables \
# 	    --recursive
# TODO: check length of chunks, if too small, what should we do?
