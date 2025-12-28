import os
import tempfile
import uuid
import mimetypes
import json
import sys
import asyncio
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.semconv_ai import GenAISystem
from opentelemetry.semconv_ai import SpanAttributes as AISpanAttributes
from opentelemetry.trace import Status, StatusCode

# from finrag.chunking import DoclingHybridChunker
from finrag.dataclasses import TopChunk
from finrag.generation_controls import (
    GenerationSettings,
    default_mode,
    list_generation_presets,
    resolve_generation_settings,
)
from finrag.llm_clients import get_llm_client
from finrag.context_support import apply_context_strategy, context_builder_from_metadata
from finrag.qa import build_draft_prompt, build_refine_prompt
from finrag.retriever import (
    CrossEncoderReranker,
    QdrantHybridRetriever,
    MilvusContextualRetriever,
    build_milvus_embedding_functions,
)
from finrag.streaming import (
    TextDeltaBatcher,
    iter_chat_deltas,
    ndjson_bytes,
    stream_chunks_max,
    stream_chunks_preview_chars,
    stream_draft_enabled,
)
from finrag.telemetry import setup_opentelemetry


# -------------------------------------------------------------------
# RAG service (ingestion + 2-stage QA)
# -------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    mode: str | None = None

    # Optional overrides (use mode preset when omitted).
    top_k_retrieve: int | None = None
    top_k_rerank: int | None = None
    draft_max_tokens: int | None = None
    final_max_tokens: int | None = None
    enable_rerank: bool | None = None
    enable_refine: bool | None = None


class QueryStreamRequest(QueryRequest):
    request_id: str | None = None


class QueryResponse(BaseModel):
    draft_answer: str
    final_answer: str
    top_chunks: list[TopChunk]


def _resolve_generation(req: QueryRequest) -> GenerationSettings:
    return resolve_generation_settings(
        mode=req.mode,
        top_k_retrieve=req.top_k_retrieve,
        top_k_rerank=req.top_k_rerank,
        draft_max_tokens=req.draft_max_tokens,
        final_max_tokens=req.final_max_tokens,
        enable_rerank=req.enable_rerank,
        enable_refine=req.enable_refine,
    )


def _request_with_resolved_settings(req: QueryRequest, settings: GenerationSettings) -> QueryRequest:
    return QueryRequest(
        question=req.question,
        mode=settings.mode,
        top_k_retrieve=settings.top_k_retrieve,
        top_k_rerank=settings.top_k_rerank,
        draft_max_tokens=settings.draft_max_tokens,
        final_max_tokens=settings.final_max_tokens,
        enable_rerank=settings.enable_rerank,
        enable_refine=settings.enable_refine,
    )


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


def _genai_system() -> str | None:
    provider = _llm_provider_name()
    if provider == "openai":
        return GenAISystem.OPENAI.value
    if provider == "mistral":
        # semconv-ai uses `MISTRALAI` (not `MISTRAL`) in current releases.
        return GenAISystem.MISTRALAI.value
    return None


def _question_fingerprint(question: str) -> str:
    # Keep cardinality low and avoid shipping user text by default.
    question = (question or "").strip()
    if not question:
        return ""
    return uuid.uuid5(uuid.NAMESPACE_URL, question).hex


def _maybe_set_otel_question(span, question: str) -> None:
    if not span.is_recording():
        return
    if not _env_bool("FINRAG_OTEL_INCLUDE_QUESTION", default=True):
        return
    question = (question or "").strip()
    if not question:
        return
    span.set_attribute("finrag.request.question", question[:4096])


def _llm_for_embeddings():
    provider = os.getenv("LLM_PROVIDER")
    if _llm_provider_name() == "openai":
        return get_llm_client(
            provider=provider,
            base_url=(os.getenv("OPENAI_EMBED_BASE_URL") or None),
            embed_model=_llm_embed_model() or "text-embedding-3-large",
        )
    embed_model = _llm_embed_model()
    return (
        get_llm_client(provider=provider, embed_model=embed_model) if embed_model else get_llm_client(provider=provider)
    )


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
    retriever = QdrantHybridRetriever(
        llm_client=_llm_for_embeddings(),
        storage_path=storage_path,
        bm25_path=bm25_path,
        context_builder=context_builder,
        context_metadata_key=context_key,
    )
    logger.info(
        f"Using Qdrant retriever with storage path: {storage_path}, bm25_path: {bm25_path}"
    )
    return retriever


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

    logger.info(
        f"Using Milvus retriever with collection: {collection_name}, sparse={use_sparse}, "
        f"uri={milvus_uri}"
    )
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


def build_reranker() -> CrossEncoderReranker:
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
    logger.info(f"Using reranker model: {reranker_model}")
    return CrossEncoderReranker(model_name=reranker_model)

class RAGService:
    def __init__(self, storage_path: str | None):
        self.llm = _llm_for_chat()
        logger.info(f"Using LLM chat model: {self.llm.chat_model}")
        self.retriever = build_retriever(storage_path)
        self.reranker = build_reranker()
        self._context_strategy, self._context_window, self._context_key = _context_config()

        # Two chunkers: with and without Mistral OCR
        # self.chunker_ocr = DoclingHybridChunker(use_mistral_ocr=True)
        # self.chunker_pdf = DoclingHybridChunker(use_mistral_ocr=False)

    @staticmethod
    def _chunk_metadata_for_ui(meta: dict | None) -> dict | None:
        if not isinstance(meta, dict):
            return None

        def _is_json_scalar(v):
            return v is None or isinstance(v, (str, int, float, bool))

        out: dict[str, object] = {}

        doc = meta.get("doc")
        if isinstance(doc, dict):
            keep = {}
            for k in (
                "company",
                "ticker",
                "cik",
                "filing_type",
                "filing_date",
                "period_end_date",
                "filing_quarter",
                "filing_quarter_basis",
            ):
                v = doc.get(k)
                if _is_json_scalar(v):
                    keep[k] = v
            if keep:
                out["doc"] = keep

        summary = meta.get("summary")
        if isinstance(summary, str) and summary.strip():
            out["summary"] = summary.strip()

        return out or None

    def _serialize_top_chunks(self, reranked) -> list[TopChunk]:
        return [
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
                metadata=self._chunk_metadata_for_ui(sc.chunk.metadata),
            )
            for sc in reranked
        ]

    def ingest_document(self, path: str, use_mistral_ocr: bool) -> str:
        """
        Ingest a single PDF at `path` using either:
        - Mistral OCR -> Markdown -> Docling -> HybridChunker
        - Direct Docling PDF parsing -> HybridChunker
        """
        raise RuntimeError("On-the-fly ingestion is disabled for now. Use batch ingestion script.")

        if isinstance(self.retriever, MilvusContextualRetriever) and self.retriever.use_sparse:
            if self.retriever.uses_bm25:
                raise RuntimeError(
                    "Online ingestion with Milvus+BM25 is disabled. "
                    "Rebuild BM25 and reindex in batch to keep sparse vectors consistent."
                )
        doc_id = str(uuid.uuid4())
        chunker = self.chunker_ocr if use_mistral_ocr else self.chunker_pdf

        # TODO: add logic from `process_html_to_markdown.py`

        docling_chunks = chunker.chunk_document(path, doc_id=doc_id)
        apply_context_strategy(
            docling_chunks,
            strategy=self._context_strategy,
            neighbor_window=self._context_window,
            metadata_key=self._context_key,
            llm_for_context=self.llm,
        )
        self.retriever.index(docling_chunks)
        return doc_id

    def answer_question(
        self,
        question: str,
        settings: GenerationSettings,
    ) -> QueryResponse:
        tracer = trace.get_tracer(__name__)

        backend = "milvus" if isinstance(self.retriever, MilvusContextualRetriever) else "qdrant"
        system = _genai_system()
        model = _llm_chat_model()

        try:
            with tracer.start_as_current_span("finrag.retrieve") as span:
                span.set_attribute(AISpanAttributes.VECTOR_DB_VENDOR, backend)
                span.set_attribute(AISpanAttributes.VECTOR_DB_OPERATION, "retrieve_hybrid")
                span.set_attribute(AISpanAttributes.VECTOR_DB_QUERY_TOP_K, int(settings.top_k_retrieve))
                hybrid = self.retriever.retrieve_hybrid(
                    question,
                    top_k_semantic=settings.top_k_retrieve,
                    top_k_bm25=settings.top_k_retrieve,
                    top_k_final=settings.top_k_retrieve,
                )
                span.set_attribute("finrag.retrieve.count", len(hybrid))

            with tracer.start_as_current_span("finrag.rerank") as span:
                span.set_attribute("finrag.rerank.enabled", bool(settings.enable_rerank))
                span.set_attribute("finrag.rerank.top_k", int(settings.top_k_rerank))
                if settings.enable_rerank:
                    reranked = self.reranker.rerank(question, hybrid, top_k=settings.top_k_rerank)
                else:
                    reranked = hybrid[: settings.top_k_rerank]
                span.set_attribute("finrag.rerank.count", len(reranked))

            draft_prompt = build_draft_prompt(
                question,
                reranked,
                draft_max_tokens=settings.draft_max_tokens,
                answer_style=settings.answer_style,
            )
            with tracer.start_as_current_span("finrag.llm.draft") as span:
                if system:
                    span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
                if model:
                    span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)
                span.set_attribute(AISpanAttributes.LLM_REQUEST_MAX_TOKENS, int(settings.draft_max_tokens))
                span.set_attribute(AISpanAttributes.LLM_REQUEST_TEMPERATURE, float(settings.draft_temperature))
                span.set_attribute(AISpanAttributes.LLM_REQUEST_TYPE, "chat")
                span.set_attribute(AISpanAttributes.LLM_IS_STREAMING, False)
                span.set_attribute("finrag.prompt.messages", len(draft_prompt))
                draft = self.llm.chat(draft_prompt, temperature=settings.draft_temperature)  # type: ignore[arg-type]
                span.set_attribute("finrag.draft.chars", len(draft))

            final = draft
            if settings.enable_refine:
                refine_prompt = build_refine_prompt(
                    question,
                    draft,
                    reranked,
                    final_max_tokens=settings.final_max_tokens,
                    answer_style=settings.answer_style,
                )
                with tracer.start_as_current_span("finrag.llm.final") as span:
                    if system:
                        span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
                    if model:
                        span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)
                    span.set_attribute(AISpanAttributes.LLM_REQUEST_MAX_TOKENS, int(settings.final_max_tokens))
                    span.set_attribute(AISpanAttributes.LLM_REQUEST_TEMPERATURE, 0.0)
                    span.set_attribute(AISpanAttributes.LLM_REQUEST_TYPE, "chat")
                    span.set_attribute(AISpanAttributes.LLM_IS_STREAMING, False)
                    span.set_attribute("finrag.prompt.messages", len(refine_prompt))
                    final = self.llm.chat(refine_prompt, temperature=0.0)  # type: ignore[arg-type]
                    span.set_attribute("finrag.final.chars", len(final))

            return QueryResponse(
                draft_answer=draft, final_answer=final, top_chunks=self._serialize_top_chunks(reranked)
            )
        except Exception as exc:  # noqa: BLE001
            span = trace.get_current_span()
            if span.is_recording():
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


# -------------------------------------------------------------------
# FastAPI wiring
# -------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]
log_path = _setup_logging(project_root)
logger.debug("Project root: %s", project_root)
logger.info("Starting RAG service; logs at: %s", log_path)

app = FastAPI(title="RAG Demo (Mistral + Docling + Qdrant)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down for real deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_opentelemetry(app)

rag_service = RAGService(storage_path=os.getenv("QDRANT_STORAGE_PATH"))

_CANCEL_LOCK = threading.Lock()
_CANCEL_EVENTS: dict[str, threading.Event] = {}


def _register_cancel_event(request_id: str) -> threading.Event:
    with _CANCEL_LOCK:
        evt = _CANCEL_EVENTS.get(request_id)
        if evt is None:
            evt = threading.Event()
            _CANCEL_EVENTS[request_id] = evt
        return evt


def _cancel_request(request_id: str) -> bool:
    with _CANCEL_LOCK:
        evt = _CANCEL_EVENTS.get(request_id)
    if evt is None:
        return False
    evt.set()
    return True


def _cleanup_cancel_event(request_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(request_id, None)


class CancelRequest(BaseModel):
    request_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/generation_presets")
def generation_presets():
    return {
        "default_mode": default_mode(),
        "presets": [p.to_public_dict() for p in list_generation_presets()],
    }


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
    # TODO: explore adding support for multi-turn Q&A
    settings = _resolve_generation(req)
    req_resolved = _request_with_resolved_settings(req, settings)

    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("finrag.request.question.len", len((req.question or "").strip()))
        span.set_attribute("finrag.request.question.fp", _question_fingerprint(req.question))
        _maybe_set_otel_question(span, req.question)
        span.set_attribute("finrag.query.mode", settings.mode)
        span.set_attribute("finrag.query.enable_rerank", bool(settings.enable_rerank))
        span.set_attribute("finrag.query.enable_refine", bool(settings.enable_refine))
        span.set_attribute("finrag.query.answer_style", settings.answer_style)
        span.set_attribute("finrag.query.top_k_retrieve", int(settings.top_k_retrieve))
        span.set_attribute("finrag.query.top_k_rerank", int(settings.top_k_rerank))
        span.set_attribute("finrag.query.draft_max_tokens", int(settings.draft_max_tokens))
        span.set_attribute("finrag.query.final_max_tokens", int(settings.final_max_tokens))
        if system := _genai_system():
            span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
        if model := _llm_chat_model():
            span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)

    try:
        result = rag_service.answer_question(question=req.question, settings=settings)
        _append_history(req=req_resolved, res=result)
        return result
    except Exception as exc:  # noqa: BLE001
        if span.is_recording():
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise


def _stream_chunk_dict(sc, *, preview_chars: int, text_chars: int) -> dict:
    text = (sc.chunk.text or "").strip()
    preview = text[:preview_chars] if preview_chars > 0 else ""
    chunk_text = text[:text_chars] if text_chars > 0 else ""
    return {
        "chunk_id": sc.chunk.id,
        "doc_id": sc.chunk.doc_id,
        "page_no": sc.chunk.page_no,
        "headings": sc.chunk.headings,
        "score": sc.score,
        "preview": preview,
        "source": sc.chunk.source,
        "text": chunk_text,
        "metadata": RAGService._chunk_metadata_for_ui(sc.chunk.metadata),
    }


@app.post("/cancel")
def cancel(req: CancelRequest):
    ok = _cancel_request((req.request_id or "").strip())
    return {"status": "ok" if ok else "not_found"}


# TODO: duplicate logic with non-streaming / rethink structure
# everytime we make a change to the answering logic, we need to update both places, not good
@app.post("/query_stream")
async def query_docs_stream(req: QueryStreamRequest, request: Request):
    request_id = (req.request_id or "").strip() or str(uuid.uuid4())
    base_req = QueryRequest(**req.dict(exclude={"request_id"}))
    settings = _resolve_generation(base_req)
    req_resolved = _request_with_resolved_settings(base_req, settings)
    cancel_evt = _register_cancel_event(request_id)
    started_ms = int(time.time() * 1000)
    parent_ctx = otel_context.get_current()
    tracer = trace.get_tracer(__name__)

    preview_chars = max(0, stream_chunks_preview_chars())
    max_chunks = max(0, stream_chunks_max())
    try:
        text_chars = int((os.getenv("FINRAG_STREAM_CHUNKS_TEXT_CHARS", "1000") or "1000").strip())
    except ValueError:
        text_chars = 1000
    text_chars = max(0, text_chars)

    async def gen():
        full_draft = ""
        full_final = ""
        cancelled = False

        token = otel_context.attach(parent_ctx)
        span = tracer.start_span("finrag.query_stream")
        span_ctx = trace.set_span_in_context(span)
        span_token = otel_context.attach(span_ctx)

        def is_cancelled() -> bool:
            return cancel_evt.is_set()

        def set_cancelled() -> None:
            cancel_evt.set()

        try:
            if span.is_recording():
                span.set_attribute("finrag.request_id", request_id)
                span.set_attribute("finrag.request.question.len", len((req.question or "").strip()))
                span.set_attribute("finrag.request.question.fp", _question_fingerprint(req.question))
                _maybe_set_otel_question(span, req.question)
                span.set_attribute("finrag.query.mode", settings.mode)
                span.set_attribute("finrag.query.enable_rerank", bool(settings.enable_rerank))
                span.set_attribute("finrag.query.enable_refine", bool(settings.enable_refine))
                span.set_attribute("finrag.query.answer_style", settings.answer_style)
                span.set_attribute("finrag.query.top_k_retrieve", int(settings.top_k_retrieve))
                span.set_attribute("finrag.query.top_k_rerank", int(settings.top_k_rerank))
                span.set_attribute("finrag.query.draft_max_tokens", int(settings.draft_max_tokens))
                span.set_attribute("finrag.query.final_max_tokens", int(settings.final_max_tokens))
                span.set_attribute("finrag.stream.preview_chars", int(preview_chars))
                span.set_attribute("finrag.stream.text_chars", int(text_chars))
                span.set_attribute("finrag.stream.max_chunks", int(max_chunks))
                if system := _genai_system():
                    span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
                if model := _llm_chat_model():
                    span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)

            yield ndjson_bytes({"type": "start", "request_id": request_id})

            yield ndjson_bytes({"type": "status", "step": "retrieve", "message": "Retrieving chunks…"})
            with tracer.start_as_current_span("finrag.retrieve") as step_span:
                backend = "milvus" if isinstance(rag_service.retriever, MilvusContextualRetriever) else "qdrant"
                step_span.set_attribute(AISpanAttributes.VECTOR_DB_VENDOR, backend)
                step_span.set_attribute(AISpanAttributes.VECTOR_DB_OPERATION, "retrieve_hybrid")
                step_span.set_attribute(AISpanAttributes.VECTOR_DB_QUERY_TOP_K, int(settings.top_k_retrieve))
                hybrid = await asyncio.to_thread(
                    rag_service.retriever.retrieve_hybrid,
                    req.question,
                    top_k_semantic=settings.top_k_retrieve,
                    top_k_bm25=settings.top_k_retrieve,
                    top_k_final=settings.top_k_retrieve,
                )
                step_span.set_attribute("finrag.retrieve.count", len(hybrid))

            if await request.is_disconnected():
                set_cancelled()
            if is_cancelled():
                cancelled = True
                yield ndjson_bytes({"type": "cancelled", "request_id": request_id, "elapsed_ms": 0})
                return

            retrieved_payload = [
                _stream_chunk_dict(sc, preview_chars=preview_chars, text_chars=text_chars) for sc in hybrid
            ]
            if max_chunks:
                retrieved_payload = retrieved_payload[:max_chunks]
            yield ndjson_bytes({"type": "retrieved", "count": len(hybrid), "chunks": retrieved_payload})

            yield ndjson_bytes(
                {
                    "type": "status",
                    "step": "rerank",
                    "message": ("Reranking chunks…" if settings.enable_rerank else "Skipping rerank (mode preset)…"),
                }
            )
            with tracer.start_as_current_span("finrag.rerank") as step_span:
                step_span.set_attribute("finrag.rerank.enabled", bool(settings.enable_rerank))
                step_span.set_attribute("finrag.rerank.top_k", int(settings.top_k_rerank))
                if settings.enable_rerank:
                    reranked = await asyncio.to_thread(
                        rag_service.reranker.rerank, req.question, hybrid, top_k=settings.top_k_rerank
                    )
                else:
                    reranked = hybrid[: settings.top_k_rerank]
                step_span.set_attribute("finrag.rerank.count", len(reranked))

            if await request.is_disconnected():
                set_cancelled()
            if is_cancelled():
                cancelled = True
                yield ndjson_bytes({"type": "cancelled", "request_id": request_id, "elapsed_ms": 0})
                return

            reranked_payload = [
                _stream_chunk_dict(sc, preview_chars=preview_chars, text_chars=text_chars) for sc in reranked
            ]
            yield ndjson_bytes({"type": "reranked", "count": len(reranked), "chunks": reranked_payload})

            if settings.enable_refine:
                yield ndjson_bytes(
                    {"type": "status", "step": "draft", "message": "Generating draft…", "is_draft": True}
                )
                draft_prompt = build_draft_prompt(
                    req.question,
                    reranked,
                    draft_max_tokens=settings.draft_max_tokens,
                    answer_style=settings.answer_style,
                )
                with tracer.start_as_current_span("finrag.llm.draft_stream") as step_span:
                    if system := _genai_system():
                        step_span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
                    if model := _llm_chat_model():
                        step_span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_MAX_TOKENS, int(settings.draft_max_tokens))
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_TEMPERATURE, float(settings.draft_temperature))
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_TYPE, "chat")
                    step_span.set_attribute(AISpanAttributes.LLM_IS_STREAMING, bool(stream_draft_enabled()))
                    step_span.set_attribute("finrag.prompt.messages", len(draft_prompt))
                    first_delta_ms: int | None = None
                    t0 = time.monotonic()

                    if stream_draft_enabled():
                        batcher = TextDeltaBatcher.from_env()
                        async for delta in iter_chat_deltas(
                            rag_service.llm,
                            draft_prompt,  # type: ignore[arg-type]
                            temperature=settings.draft_temperature,
                            is_cancelled=is_cancelled,
                            set_cancelled=set_cancelled,
                            is_disconnected=request.is_disconnected,
                        ):
                            if first_delta_ms is None and delta:
                                first_delta_ms = int((time.monotonic() - t0) * 1000)
                                step_span.add_event("finrag.first_delta", {"finrag.ms": first_delta_ms})
                            full_draft += delta
                            batcher.add(delta)
                            out = batcher.pop_ready()
                            if out:
                                yield ndjson_bytes({"type": "draft_delta", "delta": out})
                            if is_cancelled():
                                break
                        out = batcher.pop_all()
                        if out:
                            yield ndjson_bytes({"type": "draft_delta", "delta": out})
                    else:
                        full_draft = await asyncio.to_thread(
                            rag_service.llm.chat, draft_prompt, settings.draft_temperature
                        )

                    step_span.set_attribute("finrag.draft.chars", len(full_draft))

                yield ndjson_bytes({"type": "draft_done", "chars": len(full_draft)})

                if await request.is_disconnected():
                    set_cancelled()
                if is_cancelled():
                    cancelled = True
                    yield ndjson_bytes(
                        {
                            "type": "cancelled",
                            "request_id": request_id,
                            "elapsed_ms": int(time.time() * 1000) - started_ms,
                        }
                    )
                    return

                yield ndjson_bytes(
                    {"type": "status", "step": "final", "message": "Generating final answer…", "is_draft": False}
                )
                refine_prompt = build_refine_prompt(
                    req.question,
                    full_draft,
                    reranked,
                    final_max_tokens=settings.final_max_tokens,
                    answer_style=settings.answer_style,
                )
                with tracer.start_as_current_span("finrag.llm.final_stream") as step_span:
                    if system := _genai_system():
                        step_span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
                    if model := _llm_chat_model():
                        step_span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_MAX_TOKENS, int(settings.final_max_tokens))
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_TEMPERATURE, 0.0)
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_TYPE, "chat")
                    step_span.set_attribute(AISpanAttributes.LLM_IS_STREAMING, True)
                    step_span.set_attribute("finrag.prompt.messages", len(refine_prompt))
                    first_delta_ms: int | None = None
                    t0 = time.monotonic()

                    batcher = TextDeltaBatcher.from_env()
                    async for delta in iter_chat_deltas(
                        rag_service.llm,
                        refine_prompt,  # type: ignore[arg-type]
                        temperature=0.0,
                        is_cancelled=is_cancelled,
                        set_cancelled=set_cancelled,
                        is_disconnected=request.is_disconnected,
                    ):
                        if first_delta_ms is None and delta:
                            first_delta_ms = int((time.monotonic() - t0) * 1000)
                            step_span.add_event("finrag.first_delta", {"finrag.ms": first_delta_ms})
                        full_final += delta
                        batcher.add(delta)
                        out = batcher.pop_ready()
                        if out:
                            yield ndjson_bytes({"type": "final_delta", "delta": out})
                        if is_cancelled():
                            break
                    out = batcher.pop_all()
                    if out:
                        yield ndjson_bytes({"type": "final_delta", "delta": out})
                    step_span.set_attribute("finrag.final.chars", len(full_final))
            else:
                yield ndjson_bytes({"type": "status", "step": "final", "message": "Generating answer…"})
                answer_prompt = build_draft_prompt(
                    req.question,
                    reranked,
                    draft_max_tokens=settings.draft_max_tokens,
                    answer_style=settings.answer_style,
                )
                with tracer.start_as_current_span("finrag.llm.answer_stream") as step_span:
                    if system := _genai_system():
                        step_span.set_attribute(AISpanAttributes.LLM_SYSTEM, system)
                    if model := _llm_chat_model():
                        step_span.set_attribute(AISpanAttributes.LLM_REQUEST_MODEL, model)
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_MAX_TOKENS, int(settings.draft_max_tokens))
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_TEMPERATURE, float(settings.draft_temperature))
                    step_span.set_attribute(AISpanAttributes.LLM_REQUEST_TYPE, "chat")
                    step_span.set_attribute(AISpanAttributes.LLM_IS_STREAMING, True)
                    step_span.set_attribute("finrag.prompt.messages", len(answer_prompt))

                    batcher = TextDeltaBatcher.from_env()
                    async for delta in iter_chat_deltas(
                        rag_service.llm,
                        answer_prompt,  # type: ignore[arg-type]
                        temperature=settings.draft_temperature,
                        is_cancelled=is_cancelled,
                        set_cancelled=set_cancelled,
                        is_disconnected=request.is_disconnected,
                    ):
                        full_final += delta
                        batcher.add(delta)
                        out = batcher.pop_ready()
                        if out:
                            yield ndjson_bytes({"type": "final_delta", "delta": out})
                        if is_cancelled():
                            break
                    out = batcher.pop_all()
                    if out:
                        yield ndjson_bytes({"type": "final_delta", "delta": out})
                    step_span.set_attribute("finrag.final.chars", len(full_final))
                full_draft = full_final

            if await request.is_disconnected():
                set_cancelled()
            if is_cancelled():
                cancelled = True
                yield ndjson_bytes(
                    {
                        "type": "cancelled",
                        "request_id": request_id,
                        "elapsed_ms": int(time.time() * 1000) - started_ms,
                        "draft": full_draft,
                        "final_partial": full_final,
                    }
                )
                return

            res = QueryResponse(
                draft_answer=full_draft,
                final_answer=(full_final if full_final else full_draft),
                top_chunks=rag_service._serialize_top_chunks(reranked),
            )
            _append_history(req=req_resolved, res=res)

            yield ndjson_bytes(
                {
                    "type": "done",
                    "request_id": request_id,
                    "elapsed_ms": int(time.time() * 1000) - started_ms,
                    "response": jsonable_encoder(res),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming query failed: %r", exc)
            if span.is_recording():
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            yield ndjson_bytes(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": str(exc),
                    "elapsed_ms": int(time.time() * 1000) - started_ms,
                }
            )
        finally:
            if span.is_recording():
                span.set_attribute("finrag.cancelled", bool(cancelled))
            span.end()
            otel_context.detach(span_token)
            otel_context.detach(token)
            _cleanup_cancel_event(request_id)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_roots() -> list[Path]:
    raw = os.getenv("SOURCE_ROOTS")
    if raw:
        parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
        return [Path(p).expanduser().resolve() for p in parts]
    root = _project_root()
    return [root, root / "data"]


def _resolve_local_source(path: str) -> Path:
    path = (path or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="Missing `path`")

    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = (_project_root() / p).resolve()
    else:
        p = p.resolve()

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {p}")

    allowed = _source_roots()
    if not any(p == root or p.is_relative_to(root) for root in allowed):
        raise HTTPException(
            status_code=403,
            detail=("Path is outside SOURCE_ROOTS; set SOURCE_ROOTS to a colon-separated allowlist of directories."),
        )

    return p


@app.get("/source")
def get_source(path: str = Query(..., description="Local file path or URL")):
    path = (path or "").strip()
    if path.startswith(("http://", "https://")):
        return RedirectResponse(url=path)
    p = _resolve_local_source(path)
    media_type, _enc = mimetypes.guess_type(str(p))
    return FileResponse(
        path=p, media_type=media_type or "application/octet-stream", filename=p.name, content_disposition_type="inline"
    )


def _read_text_file(path: Path, *, max_bytes: int) -> str:
    if max_bytes <= 0:
        raise HTTPException(status_code=400, detail="SOURCE_TEXT_MAX_BYTES must be > 0")
    size = path.stat().st_size
    if size > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large ({size} bytes); max is {max_bytes} bytes")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


@app.get("/source_text")
def get_source_text(path: str = Query(..., description="Local markdown/text file path")):
    p = _resolve_local_source(path)
    suffix = p.suffix.lower()
    if suffix not in {".md", ".markdown", ".txt"}:
        raise HTTPException(status_code=415, detail="Only .md/.markdown/.txt are supported for inline text viewing")

    max_bytes_raw = os.getenv("SOURCE_TEXT_MAX_BYTES", "5000000").strip()
    try:
        max_bytes = int(max_bytes_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="SOURCE_TEXT_MAX_BYTES must be an integer") from exc

    return {"path": str(p), "text": _read_text_file(p, max_bytes=max_bytes)}


class HistoryEntry(BaseModel):
    id: str
    created_at: str
    request: QueryRequest
    response: QueryResponse


def _history_path() -> Path:
    raw = os.getenv("HISTORY_PATH")
    if raw and raw.strip():
        return Path(os.path.expanduser(raw.strip())).resolve()
    return (_project_root() / "data" / "qa_history.jsonl").resolve()


def _append_history(*, req: QueryRequest, res: QueryResponse) -> None:
    if _env_bool("DISABLE_HISTORY", default=False):
        return
    entry = HistoryEntry(
        id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(), request=req, response=res
    )
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable_encoder(entry), ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - history should never break /query
        logger.warning("Failed to write history to %s: %r", path, exc)


def _read_history(*, limit: int = 50, summary: bool = False) -> list[dict]:
    limit = max(0, int(limit))
    path = _history_path()
    if limit == 0 or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in (line.strip() for line in f) if ln]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read history from %s: %r", path, exc)
        return []

    out: list[dict] = []
    for line in reversed(lines[-limit:]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if not summary:
            out.append(payload)
            continue

        # Keep the list lightweight for the UI: omit large answer text and chunks.
        req_raw = payload.get("request")
        req: dict[str, object] = req_raw if isinstance(req_raw, dict) else {}

        res_raw = payload.get("response")
        res: dict[str, object] = res_raw if isinstance(res_raw, dict) else {}

        top_chunks_raw = res.get("top_chunks")
        top_chunks: list[object] = top_chunks_raw if isinstance(top_chunks_raw, list) else []

        out.append(
            {
                "id": payload.get("id"),
                "created_at": payload.get("created_at"),
                "request": {
                    "question": req.get("question"),
                    "mode": req.get("mode"),
                },
                "response": {
                    "top_chunks_count": len(top_chunks),
                },
            }
        )
    return out


@app.get("/history")
def history(limit: int = 50, summary: bool = False):
    return {"items": _read_history(limit=limit, summary=summary), "path": str(_history_path())}


@app.get("/history_entry")
def history_entry(id: str = Query(..., description="History entry id")):
    want = (id or "").strip()
    if not want:
        raise HTTPException(status_code=400, detail="Missing `id`")

    path = _history_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="History file not found")

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in (line.strip() for line in f) if ln]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read history: {exc}") from exc

    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and str(payload.get("id") or "") == want:
            return payload

    raise HTTPException(status_code=404, detail="History entry not found")


@app.delete("/history")
def clear_history():
    path = _history_path()
    if path.exists():
        path.unlink()
    return {"status": "ok", "path": str(path)}


# -------------------------------------------------------------------
# Simple HTML frontend
# -------------------------------------------------------------------

HTML_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    # Read on request so frontend edits don't require a server restart.
    return HTML_PATH.read_text(encoding="utf-8")
