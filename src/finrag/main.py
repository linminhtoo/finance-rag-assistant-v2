import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from mistralai import Mistral
from mistralai.models.chatcompletionrequest import MessagesTypedDict
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from finrag.chunking import DoclingHybridChunker
from finrag.dataclasses import DocChunk, ScoredChunk
from finrag.utils import get_env_var

# -------------------------------------------------------------------
# Mistral client wrapper (embeddings + chat)
# -------------------------------------------------------------------

class MistralClientWrapper:
    def __init__(
        self,
        api_key_env: str = "MISTRAL_API_KEY",
        chat_model: str = "mistral-small-latest",
        embed_model: str = "mistral-embed",
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.client = Mistral(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.embed_model,
            inputs=texts,
        )
        vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vectors)

    def chat(self, messages: list[MessagesTypedDict], temperature: float = 0.1) -> str:
        res = self.client.chat.complete(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        try:
            return res.choices[0].message.content  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to get chat response: {e}") from e
    
    # not used. just experimenting.
    # def structured_chat(
    #     self,
    #     messages: list[dict[str, str]],
    #     response_model: BaseModel,
    #     temperature: float = 0.1,
    # ) -> str | None:
    #     res = self.client.chat.parse(
    #         model=self.chat_model,
    #         messages=messages,
    #         response_format=response_model,
    #         temperature=temperature,
    #     )
    #     return res.choices[0].message.content


# -------------------------------------------------------------------
# Hybrid retriever: Qdrant + BM25
# -------------------------------------------------------------------



class HybridRetriever:
    def __init__(
        self,
        mistral: MistralClientWrapper,
        storage_path: str,
        collection_name: str = "rag_chunks",
        vector_dim: int = 1024,
    ):
        # TODO: allow swap to OpenAI client
        self.mistral = mistral
        self.collection_name = collection_name

        # For a real deployment, point to external Qdrant (e.g. http://qdrant:6333)
        self.storage_path = storage_path
        self.qdrant = QdrantClient(path=storage_path)  # ":memory:" also works

        if not self.qdrant.collection_exists(collection_name):
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,
                ),
            )

        self.chunks_by_id: dict[str, DocChunk] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_chunk_ids: list[str] = []

    def index(self, chunks: list[DocChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.mistral.embed_texts(texts)

        points = []
        for emb, chunk in zip(embeddings, chunks):
            self.chunks_by_id[chunk.id] = chunk
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=emb.tolist(),
                    payload=chunk.as_payload(),
                )
            )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        # BM25
        for chunk in chunks:
            tokens = chunk.text.split()
            self._bm25_corpus.append(tokens)
            self._bm25_chunk_ids.append(chunk.id)
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def _semantic_search(self, query: str, top_k: int = 20) -> list[tuple]:
        q_emb = self.mistral.embed_texts([query])[0]
        hits = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_emb.tolist(),
            limit=top_k,
            with_payload=False,
        )
        return [(str(pt.id), float(pt.score)) for pt in hits.points]

    def _bm25_search(self, query: str, top_k: int = 20) -> list[tuple]:
        if self._bm25 is None:
            return []
        tokens = query.split()
        scores = self._bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [
            (self._bm25_chunk_ids[i], float(scores[i]))
            for i in idxs
            if scores[i] > 0
        ]

    def retrieve_hybrid(
        self,
        query: str,
        top_k_semantic: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 20,
        alpha: float = 0.6,
    ) -> list[ScoredChunk]:
        sem_results = self._semantic_search(query, top_k_semantic)
        bm25_results = self._bm25_search(query, top_k_bm25)

        def normalize(results):
            if not results:
                return {}
            vals = np.array([s for _, s in results], dtype=np.float32)
            min_v, max_v = float(vals.min()), float(vals.max())
            if max_v - min_v < 1e-9:
                return {cid: 1.0 for cid, _ in results}
            return {cid: (float(s) - min_v) / (max_v - min_v) for cid, s in results}

        sem_norm = normalize(sem_results)
        bm25_norm = normalize(bm25_results)

        combined: dict[str, float] = {}
        for cid, s in sem_norm.items():
            combined[cid] = combined.get(cid, 0.0) + alpha * s
        for cid, s in bm25_norm.items():
            combined[cid] = combined.get(cid, 0.0) + (1 - alpha) * s

        sorted_ids = sorted(
            combined.items(), key=lambda kv: kv[1], reverse=True
        )[:top_k_final]

        out: list[ScoredChunk] = []
        for cid, score in sorted_ids:
            chunk = self.chunks_by_id[cid]
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    source="hybrid",
                )
            )
        return out


# -------------------------------------------------------------------
# Cross-encoder reranker
# -------------------------------------------------------------------

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 10
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs).tolist()
        rescored = []
        for cand, score in zip(candidates, scores):
            rescored.append(
                ScoredChunk(
                    chunk=cand.chunk,
                    score=float(score),
                    source="reranker",
                )
            )
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]


# -------------------------------------------------------------------
# RAG service (ingestion + 2-stage QA)
# -------------------------------------------------------------------

class RAGService:
    def __init__(self, storage_path: str):
        self.mistral = MistralClientWrapper()
        self.retriever = HybridRetriever(self.mistral, storage_path=storage_path)
        self.reranker = CrossEncoderReranker()

        # Two chunkers: with and without Mistral OCR
        self.chunker_ocr = DoclingHybridChunker(
            use_mistral_ocr=True,
        )
        self.chunker_pdf = DoclingHybridChunker(
            use_mistral_ocr=False,
        )

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

    def _build_context(
        self, chunks: list[ScoredChunk], max_tokens: int
    ) -> str:
        budget_chars = max_tokens * 4
        parts = []
        used = 0
        for sc in chunks:
            meta = (
                f"[doc={sc.chunk.doc_id} "
                # f"page={sc.chunk.page_no} "
                f"headings={'; '.join(sc.chunk.headings)}]"
            )
            text = sc.chunk.text.strip()
            block = f"{meta}\n{text}\n"
            if used + len(block) > budget_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def answer_question(
        self,
        question: str,
        top_k_retrieve: int = 30,
        top_k_rerank: int = 8,
    ) -> dict[str, Any]:
        # 1) Hybrid retrieve
        hybrid = self.retriever.retrieve_hybrid(
            question,
            top_k_semantic=top_k_retrieve,
            top_k_bm25=top_k_retrieve,
            top_k_final=top_k_retrieve,
        )

        # 2) Cross-encoder rerank
        reranked = self.reranker.rerank(question, hybrid, top_k=top_k_rerank)

        # 3) Stage 1: draft
        ctx1 = self._build_context(reranked, max_tokens=900)
        # NOTE: MessagesTypedDict type is specific to mistral. 
        # need to handle OpenAI differently if used.
        draft_prompt: list[MessagesTypedDict] = [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant answering questions over government "
                    "policy PDFs. Always stay grounded in the provided context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{ctx1}\n\n"
                    "Answer concisely and list which [doc=..., page=...] segments you used."
                ),
            },
        ]
        draft = self.mistral.chat(draft_prompt, temperature=0.1)

        # 4) Stage 2: refine
        ctx2 = self._build_context(reranked, max_tokens=1500)
        # NOTE: MessagesTypedDict type is specific to mistral. 
        # need to handle OpenAI differently if used.
        refine_prompt: list[MessagesTypedDict] = [
            {
                "role": "system",
                "content": (
                    "You are a senior policy analyst. You must:\n"
                    "1) check the draft answer against the context;\n"
                    "2) fix hallucinations;\n"
                    "3) clearly state if context is insufficient."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{question}\n\n"
                    f"Draft answer:\n{draft}\n\n"
                    f"Context:\n{ctx2}\n\n"
                    "Now write a refined answer. Start with a short paragraph, "
                    "then add a 'Sources' section referencing [doc=..., page=...]."
                ),
            },
        ]
        final = self.mistral.chat(refine_prompt, temperature=0.0)

        # Serialize top chunks for the frontend
        top_chunks = [
            {
                "doc_id": sc.chunk.doc_id,
                # "page_no": sc.chunk.page_no,
                "headings": sc.chunk.headings,
                "score": sc.score,
                "preview": sc.chunk.text[:300],
            }
            for sc in reranked
        ]

        return {
            "draft_answer": draft,
            "final_answer": final,
            "top_chunks": top_chunks,
        }


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


class QueryRequest(BaseModel):
    question: str
    top_k_retrieve: int = 30
    top_k_rerank: int = 8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    use_mistral_ocr: bool = Form(False),
):
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
        question=req.question,
        top_k_retrieve=req.top_k_retrieve,
        top_k_rerank=req.top_k_rerank,
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


# If you want to run with `python app.py`
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
