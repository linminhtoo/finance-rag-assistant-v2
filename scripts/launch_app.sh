#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# NOTE: rmbr to set OPENAI_EMBED_BASE_URL and OPENAI_CHAT_BASE_URL in .env
export OPENAI_CHAT_MODEL="Qwen/Qwen3-VL-32B-Instruct-FP8"
export OPENAI_EMBED_MODEL="BAAI/bge-m3"

export MILVUS_COLLECTION_NAME="with_llm_context_27_12_25"
export MILVUS_DENSE_EMBEDDING="llm"
export MILVUS_SPARSE_EMBEDDING="bm25"
export CONTEXT_STRATEGY="neighbors"
export CONTEXT_WINDOW=8

# FIXME: MILVUS_URI only accepts https:// , so we must use MILVUS_PATH for local storage
export MILVUS_PATH="/home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128/milvus.db"
export BM25_PATH="/home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128/bm25.pkl"

# optionally set HISTORY_PATH / DISABLE_HISTORY=true / SOURCE_ROOTS

PYTHONPATH=src uvicorn finrag.main:app --host 0.0.0.0 --port 8234 --reload
