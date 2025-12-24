#!/bin/bash
set -euo pipefail

# NOTE: script does logging set up internally. we do NOT need to `tee`
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python3 -m scripts.chunk \
	    --markdown-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings/processed_markdown \
	    --output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings/chunked_v1 \
	    --preprocess-markdown-tables \
	    --recursive
# TODO: check length of chunks, if too small, what should we do?
