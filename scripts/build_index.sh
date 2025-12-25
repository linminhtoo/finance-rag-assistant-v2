#!/bin/bash
set -euo pipefail

# NOTE: script does logging set up internally. we do NOT need to `tee`
# NOTE: we can host an embedding model via vLLM, then set --llm-provider openai
# and specify --embedding-model to point to our hosted model.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

python3 -m scripts.build_index \
	--ingest-output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings/chunked_v1 \
	--storage-path /home/mlin/repos/z_scratch/financial-rag/outputs/qdrant_sec \
	--collection-name chunks_v1_25_12_25 \
	--llm-provider openai \
	--embedding-model BAAI/bge-m3 \
	--overwrite-collection
