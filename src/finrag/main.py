import os
import tempfile
import uuid
import mimetypes
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from loguru import logger
from pydantic import BaseModel

from finrag.chunking import DoclingHybridChunker
from finrag.dataclasses import TopChunk
from finrag.llm_clients import get_llm_client
from finrag.context_support import apply_context_strategy, context_builder_from_metadata
from finrag.qa import answer_question_two_stage
from finrag.retriever import CrossEncoderReranker, QdrantHybridRetriever, MilvusContextualRetriever, build_milvus_embedding_functions


# -------------------------------------------------------------------
# RAG service (ingestion + 2-stage QA)
# -------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    top_k_retrieve: int = 30
    top_k_rerank: int = 8
    draft_max_tokens: int = 65_536
    final_max_tokens: int = 32_768


class QueryResponse(BaseModel):
    draft_answer: str
    final_answer: str
    top_chunks: list[TopChunk]


def _setup_logging(project_root: Path) -> Path:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"main_app_{ts}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="DEBUG")
    return log_path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _llm_provider_name() -> str:
    return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def _llm_chat_model() -> str | None:
    provider = _llm_provider_name()
    if provider == "openai":
        return (os.getenv("OPENAI_CHAT_MODEL") or os.getenv("CHAT_MODEL") or "").strip() or None
    if provider == "mistral":
        return (os.getenv("MISTRAL_CHAT_MODEL") or os.getenv("CHAT_MODEL") or "").strip() or None
    return (os.getenv("CHAT_MODEL") or "").strip() or None


def _llm_embed_model() -> str | None:
    provider = _llm_provider_name()
    if provider == "openai":
        return (os.getenv("OPENAI_EMBED_MODEL") or os.getenv("EMBED_MODEL") or "").strip() or None
    if provider == "mistral":
        return (os.getenv("MISTRAL_EMBED_MODEL") or os.getenv("EMBED_MODEL") or "").strip() or None
    if provider == "fastembed":
        return (os.getenv("FASTEMBED_EMBED_MODEL") or os.getenv("EMBED_MODEL") or "").strip() or None
    return (os.getenv("EMBED_MODEL") or "").strip() or None


def _llm_for_embeddings():
    provider = os.getenv("LLM_PROVIDER")
    if _llm_provider_name() == "openai":
        return get_llm_client(
            provider=provider,
            base_url=(os.getenv("OPENAI_EMBED_BASE_URL") or None),
            embed_model=_llm_embed_model() or "text-embedding-3-large",
        )
    embed_model = _llm_embed_model()
    return get_llm_client(provider=provider, embed_model=embed_model) if embed_model else get_llm_client(provider=provider)


def _llm_for_chat():
    provider = os.getenv("LLM_PROVIDER")
    langsmith_trace = False
    if os.environ.get("LANGSMITH_TRACING", "false").lower() == "true":
        langsmith_trace = True
    
    if _llm_provider_name() == "openai":
        return get_llm_client(
            provider=provider,
            base_url=(os.getenv("OPENAI_CHAT_BASE_URL") or None),
            chat_model=_llm_chat_model() or "gpt-4o-mini",
            langsmith_trace=langsmith_trace,
        )
    
    if langsmith_trace:
        logger.warning("LANGSMITH_TRACING is only supported for OpenAI provider at this time.")
    chat_model = _llm_chat_model()
    return get_llm_client(provider=provider, chat_model=chat_model) if chat_model else get_llm_client(provider=provider)


def _context_config() -> tuple[str, int, str]:
    strategy = os.getenv("CONTEXT_STRATEGY", "none").strip().lower()
    window_raw = os.getenv("CONTEXT_WINDOW", "1")
    try:
        window = int(window_raw)
    except ValueError as exc:
        raise RuntimeError("CONTEXT_WINDOW must be an integer") from exc
    metadata_key = os.getenv("CONTEXT_METADATA_KEY", "context").strip() or "context"
    return strategy, window, metadata_key


def build_hybrid_retriever(storage_path: str) -> QdrantHybridRetriever:
    """
    Build the default Qdrant-backed hybrid retriever.

    Parameters
    ----------
    storage_path : str
        Filesystem path for Qdrant storage.

    Returns
    -------
    HybridRetriever
        Configured retriever instance.
    """

    bm25_path = os.getenv("BM25_PATH")
    if bm25_path:
        bm25_path = os.path.expanduser(bm25_path)
    _, _, context_key = _context_config()
    context_builder = context_builder_from_metadata(key=context_key)
    return QdrantHybridRetriever(
        llm_client=_llm_for_embeddings(),
        storage_path=storage_path,
        bm25_path=bm25_path,
        context_builder=context_builder,
        context_metadata_key=context_key,
    )


def build_milvus_retriever() -> MilvusContextualRetriever:
    """
    Build the Milvus-backed retriever using environment configuration.

    Returns
    -------
    MilvusContextualRetriever
        Configured retriever instance.
    """

    project_root = Path(__file__).resolve().parents[2]
    
    # NOTE: MILVUS_URI only accepts http[s]://
    # filepaths should be provided via MILVUS_PATH for local storage
    milvus_uri = os.getenv("MILVUS_URI") or os.getenv("MILVUS_PATH") or str(project_root / "data" / "milvus.db")
    if "://" not in milvus_uri:
        milvus_uri = os.path.expanduser(milvus_uri)
    logger.info(f"Using {milvus_uri=}")

    collection_name = os.getenv("MILVUS_COLLECTION_NAME") or "finrag_milvus_collection"
    
    use_sparse = _env_bool("MILVUS_USE_SPARSE", default=True)
    sparse_kind = os.getenv("MILVUS_SPARSE_EMBEDDING", "bm25").strip().lower()
    if sparse_kind == "none":
        use_sparse = False

    bm25_path = os.getenv("BM25_PATH")
    if bm25_path:
        bm25_path = os.path.expanduser(bm25_path)
    
    dense_kind = os.getenv("MILVUS_DENSE_EMBEDDING", "llm").strip().lower()

    _, _, context_key = _context_config()
    dense_fn, sparse_fn = build_milvus_embedding_functions(
        llm_client_for_dense=_llm_for_embeddings(),
        dense_kind=dense_kind,
        sparse_kind=sparse_kind,
        use_sparse=use_sparse,
    )
    context_builder = context_builder_from_metadata(key=context_key)
    retriever = MilvusContextualRetriever(
        uri=milvus_uri,
        collection_name=collection_name,
        use_sparse=use_sparse,
        bm25_path=bm25_path,
        dense_embedding_function=dense_fn,
        sparse_embedding_function=sparse_fn,
        context_builder=context_builder,
        context_metadata_key=context_key,
    )

    if use_sparse and retriever.uses_bm25:
        if bm25_path is None:
            raise RuntimeError("MILVUS_USE_SPARSE=true requires MILVUS_BM25_PATH (or BM25_PATH).")
        if not Path(bm25_path).exists():
            raise RuntimeError(f"BM25 parameters not found at: {bm25_path}")
        retriever.load_bm25(bm25_path)

    return retriever


def build_retriever(storage_path: str | None) -> QdrantHybridRetriever | MilvusContextualRetriever:
    """
    Build a retriever based on the configured backend.

    Parameters
    ----------
    storage_path : str | None
        Qdrant storage path. Required when RETRIEVER_BACKEND=qdrant.

    Returns
    -------
    QdrantHybridRetriever | MilvusContextualRetriever
        Configured retriever instance.
    """

    backend = os.getenv("RETRIEVER_BACKEND", "qdrant").strip().lower()
    if backend == "milvus":
        return build_milvus_retriever()
    if storage_path is None:
        raise RuntimeError("QDRANT_STORAGE_PATH is required when RETRIEVER_BACKEND=qdrant.")
    return build_hybrid_retriever(storage_path)


class RAGService:
    def __init__(self, storage_path: str):
        self.llm = get_llm_client()
        self.retriever = build_hybrid_retriever(storage_path)
        self.reranker = CrossEncoderReranker()

        # Two chunkers: with and without Mistral OCR
        self.chunker_ocr = DoclingHybridChunker(use_mistral_ocr=True)
        self.chunker_pdf = DoclingHybridChunker(use_mistral_ocr=False)

    def ingest_document(self, path: str, use_mistral_ocr: bool) -> str:
        """
        Ingest a single PDF at `path` using either:
        - Mistral OCR -> Markdown -> Docling -> HybridChunker
        - Direct Docling PDF parsing -> HybridChunker
        """
        doc_id = str(uuid.uuid4())
        chunker = self.chunker_ocr if use_mistral_ocr else self.chunker_pdf

        docling_chunks = chunker.chunk_document(path, doc_id=doc_id)
        self.retriever.index(docling_chunks)
        return doc_id

    def answer_question(self, question: str, top_k_retrieve: int = 30, top_k_rerank: int = 8) -> QueryResponse:
        # 1) Hybrid retrieve
        hybrid = self.retriever.retrieve_hybrid(
            question, top_k_semantic=top_k_retrieve, top_k_bm25=top_k_retrieve, top_k_final=top_k_retrieve
        )

        # 2) Cross-encoder rerank
        reranked = self.reranker.rerank(question, hybrid, top_k=top_k_rerank)

        draft, final = answer_question_two_stage(
            self.llm, question, reranked, draft_max_tokens=900, final_max_tokens=1500
        )

        # Serialize top chunks for the frontend
        top_chunks = [
            TopChunk(
                chunk_id=sc.chunk.id,
                doc_id=sc.chunk.doc_id,
                page_no=sc.chunk.page_no,
                headings=sc.chunk.headings,
                score=sc.score,
                preview=sc.chunk.text[:300],
                source=sc.chunk.source,
                text=sc.chunk.text,
                context=(
                    str((sc.chunk.metadata or {}).get(self._context_key))
                    if (sc.chunk.metadata or {}).get(self._context_key) is not None
                    else None
                ),
            )
            for sc in reranked
        ]

        return QueryResponse(draft_answer=draft, final_answer=final, top_chunks=top_chunks)


# -------------------------------------------------------------------
# FastAPI wiring
# -------------------------------------------------------------------

app = FastAPI(title="RAG Demo (Mistral + Docling + Qdrant)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down for real deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: add Langsmith tracing
rag_service = RAGService(storage_path=get_env_var("QDRANT_STORAGE_PATH"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...), use_mistral_ocr: bool = Form(False)):
    # Save uploaded file to a temp path
    filename = file.filename
    if filename is None:
        raise ValueError("Uploaded file must have a filename")
    suffix = os.path.splitext(filename)[-1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        doc_id = rag_service.ingest_document(tmp_path, use_mistral_ocr=use_mistral_ocr)
    finally:
        # optional: keep PDFs for audit; for now, delete
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return {"doc_id": doc_id}


@app.post("/query")
async def query_docs(req: QueryRequest):
    result = rag_service.answer_question(
        question=req.question, top_k_retrieve=req.top_k_retrieve, top_k_rerank=req.top_k_rerank
    )
    return result


# -------------------------------------------------------------------
# Simple HTML frontend
# -------------------------------------------------------------------

HTML_PATH = Path(__file__).parent / "static" / "index.html"
with open(HTML_PATH, "r") as f:
    HTML_PAGE = f.read()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
