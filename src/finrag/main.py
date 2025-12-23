import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from finrag.chunking import DoclingHybridChunker
from finrag.dataclasses import TopChunk
from finrag.llm_clients import get_llm_client
from finrag.qa import answer_question_two_stage
from finrag.retriever import CrossEncoderReranker, HybridRetriever
from finrag.utils import get_env_var


# -------------------------------------------------------------------
# RAG service (ingestion + 2-stage QA)
# -------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    top_k_retrieve: int = 30
    top_k_rerank: int = 8


class QueryResponse(BaseModel):
    draft_answer: str
    final_answer: str
    top_chunks: list[TopChunk]


def build_hybrid_retriever(storage_path: str) -> HybridRetriever:
    bm25_path = os.getenv("BM25_PATH")
    if bm25_path:
        bm25_path = os.path.expanduser(bm25_path)
    return HybridRetriever(get_llm_client(), storage_path=storage_path, bm25_path=bm25_path)


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
                doc_id=sc.chunk.doc_id,
                page_no=sc.chunk.page_no,
                headings=sc.chunk.headings,
                score=sc.score,
                preview=sc.chunk.text[:300],
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
