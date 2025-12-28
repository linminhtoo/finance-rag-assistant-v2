#!/bin/bash
set -euo pipefail

# NOTE: script does logging set up internally. we do NOT need to `tee`
# NOTE: we can host an embedding model via vLLM, then set --llm-provider openai
# and specify --dense-model to point to our hosted model.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# to run embedding model locally using BAAI's FlagEmbedding library
# --milvus-dense-embedding bge-m3 \

# 33 sec to fit BM25 corpus on 15k chunks (119 markdown files)
# expand existing collection
python3 -m scripts.build_index \
	--ingest-output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128 \
	--collection-name with_llm_context_27_12_25 \
	--llm-provider openai \
	--milvus-sparse "bm25" \
	--dense-model BAAI/bge-m3 \
	--dense-base-url "${OPENAI_EMBED_BASE_URL}" \
	--batch-size 128 \
	--contextual-llm-provider openai \
	--contextual-model "Qwen/Qwen3-VL-32B-Instruct-FP8" \
	--contextual-base-url "${OPENAI_CONTEXT_BASE_URL}" \
	--context "neighbors" \
	--context-window 8 \
	--context-max-concurrency 64 \
	--expand-collection

# first try -> about 60-70s per doc, but vLLM server was suboptimal (20 GB VRAM not used)
# python3 -m scripts.build_index \
# 	--ingest-output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128 \
# 	--collection-name with_llm_context_27_12_25 \
# 	--llm-provider openai \
# 	--milvus-sparse "bm25" \
# 	--dense-model BAAI/bge-m3 \
# 	--dense-base-url "http://192.168.3.184:8913/v1" \
# 	--batch-size 128 \
# 	--contextual-llm-provider openai \
# 	--contextual-model "Qwen/Qwen3-VL-32B-Instruct-FP8" \
# 	--contextual-base-url "http://192.168.3.184:8993/v1" \
# 	--context "neighbors" \
# 	--context-window 8 \
# 	--context-max-concurrency 64


# python3 -m scripts.build_index \
# 	--ingest-output-dir /home/mlin/repos/z_scratch/financial-rag/data/sec_filings/chunked_v1 \
# 	--qdrant-storage-path /home/mlin/repos/z_scratch/financial-rag/outputs/qdrant_sec \
# 	--collection-name chunks_v1_25_12_25 \
# 	--llm-provider openai \
# 	--dense-model BAAI/bge-m3
