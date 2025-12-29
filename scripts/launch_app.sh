#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# NOTE: rmbr to set OPENAI_EMBED_BASE_URL and OPENAI_CHAT_BASE_URL in .env
export OPENAI_CHAT_MODEL="Qwen/Qwen3-VL-32B-Instruct-FP8"
export OPENAI_EMBED_MODEL="BAAI/bge-m3"
# export RERANKER_MODEL="BAAI/bge-reranker-v2-gemma"
export RERANKER_MODEL="BAAI/bge-reranker-v2-m3"

export MILVUS_COLLECTION_NAME="with_llm_context_27_12_25"
export MILVUS_DENSE_EMBEDDING="llm"
export MILVUS_SPARSE_EMBEDDING="bm25"
export CONTEXT_STRATEGY="neighbors"
export CONTEXT_WINDOW=8

# FIXME: MILVUS_URI only accepts https:// , so we must use MILVUS_PATH for local storage
export MILVUS_PATH="/home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128/milvus.db"
export BM25_PATH="/home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128/bm25.pkl"
export FINRAG_DOC_INDEX_PATH="/home/mlin/repos/z_scratch/financial-rag/data/sec_filings_md_v5/chunked_1024_128/doc_index.jsonl"

# optionally set HISTORY_PATH / DISABLE_HISTORY=true / SOURCE_ROOTS

# new env vars
# FINRAG_STREAM_FLUSH_TOKENS (default 24), FINRAG_STREAM_FLUSH_CHARS (default 120), FINRAG_STREAM_FLUSH_INTERVAL_MS (default 120)
# FINRAG_STREAM_DRAFT (1/0, default 1)
# FINRAG_STREAM_CHUNKS_PREVIEW_CHARS (default 260), FINRAG_STREAM_CHUNKS_TEXT_CHARS (default 1000), FINRAG_STREAM_CHUNKS_MAX (default 30)

# FINRAG_OTEL_ENABLED=true|false (defaults to on when OTEL_EXPORTER_OTLP_ENDPOINT is set)
# FINRAG_OTEL_CONSOLE=true to also print spans to stdout
# FINRAG_OTEL_INCLUDE_QUESTION=false to exclude tracking questions (will only track length + fingerprint)
export FINRAG_OTEL_CONSOLE=false
export FINRAG_OTEL_ENABLED=true

source /home/mlin/repos/z_scratch/financial-rag/.venv/bin/activate
PYTHONPATH=src uvicorn finrag.main:app --host 0.0.0.0 --port 8236
#  --reload
