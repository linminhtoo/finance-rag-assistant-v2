import uuid
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from finrag.dataclasses import DocChunk, ScoredChunk
from finrag.llm_clients import LLMClient

# -------------------------------------------------------------------
# Hybrid retriever: Qdrant + BM25
# -------------------------------------------------------------------


class HybridRetriever:
    def __init__(
        self,
        llm_client: LLMClient,
        storage_path: str,
        collection_name: str = "rag_chunks",
        vector_dim: int | None = None,
        *,
        load_existing: bool = True,
        load_batch_size: int = 512,
    ):
        self.llm = llm_client
        self.collection_name = collection_name

        # For a real deployment, point to external Qdrant (e.g. http://qdrant:6333)
        self.storage_path = storage_path
        self.qdrant = QdrantClient(path=storage_path)  # ":memory:" also works

        self.vector_dim = vector_dim
        if self.qdrant.collection_exists(collection_name):
            info = self.qdrant.get_collection(collection_name)
            self.vector_dim = int(info.config.params.vectors.size)  # type: ignore[attr-defined]
        elif vector_dim is not None:
            self.qdrant.create_collection(
                collection_name=collection_name, vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )

        self.chunks_by_id: dict[str, DocChunk] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_chunk_ids: list[str] = []
        self._qdrant_id_by_chunk_id: dict[str, str] = {}

        if load_existing and self.qdrant.collection_exists(collection_name):
            self._load_existing(batch_size=load_batch_size)

    @staticmethod
    def _qdrant_id_for_chunk_id(chunk_id: str) -> str:
        # Local Qdrant expects UUID-like point IDs; make a deterministic UUID from chunk_id.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _chunk_from_payload(point_id: str, payload: dict[str, Any]) -> DocChunk:
        return DocChunk(
            id=str(payload.get("chunk_id") or point_id),
            doc_id=str(payload.get("doc_id") or ""),
            text=str(payload.get("text") or ""),
            page_no=payload.get("page_no"),
            headings=list(payload.get("headings") or []),
            source=str(payload.get("source") or ""),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        )

    def _load_existing(self, *, batch_size: int = 512) -> None:
        offset = None
        seen = 0
        while True:
            records, offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for rec in records:
                if rec.payload is None:
                    continue
                payload = dict(rec.payload)
                point_id = str(rec.id)
                ch = self._chunk_from_payload(point_id, payload)
                self.chunks_by_id[point_id] = ch
                self._qdrant_id_by_chunk_id[ch.id] = point_id
                text = (ch.metadata or {}).get("index_text") or ch.text
                self._bm25_corpus.append(str(text).split())
                self._bm25_chunk_ids.append(point_id)
                seen += 1
            if offset is None:
                break
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    def index(self, chunks: list[DocChunk]) -> None:
        if not chunks:
            return

        texts = [(c.metadata or {}).get("index_text") or c.text for c in chunks]
        embeddings = self.llm.embed_texts(texts)
        dim = int(embeddings.shape[1])
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            self.vector_dim = dim
        elif self.vector_dim is not None and dim != self.vector_dim:
            raise ValueError(f"Embedding dim {dim} != collection dim {self.vector_dim} for {self.collection_name}")

        points = []
        for emb, chunk in zip(embeddings, chunks):
            point_id = self._qdrant_id_for_chunk_id(chunk.id)
            self.chunks_by_id[point_id] = chunk
            self._qdrant_id_by_chunk_id[chunk.id] = point_id
            points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=chunk.as_payload()))

        self.qdrant.upsert(collection_name=self.collection_name, points=points, wait=True)

        # BM25
        # TODO: should we add some preprocessing like lowercasing, removing stopwords, etc.?
        for chunk in chunks:
            text = (chunk.metadata or {}).get("index_text") or chunk.text
            tokens = text.split()
            self._bm25_corpus.append(tokens)
            self._bm25_chunk_ids.append(self._qdrant_id_by_chunk_id[chunk.id])
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def _semantic_search(self, query: str, top_k: int = 20) -> list[tuple]:
        q_emb = self.llm.embed_texts([query])[0]
        hits = self.qdrant.query_points(
            collection_name=self.collection_name, query=q_emb.tolist(), limit=top_k, with_payload=False
        )
        return [(str(pt.id), float(pt.score)) for pt in hits.points]

    def _bm25_search(self, query: str, top_k: int = 20) -> list[tuple]:
        if self._bm25 is None:
            return []
        tokens = query.split()
        scores = self._bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self._bm25_chunk_ids[i], float(scores[i])) for i in idxs if scores[i] > 0]

    def retrieve_hybrid(
        self, query: str, top_k_semantic: int = 20, top_k_bm25: int = 20, top_k_final: int = 20, alpha: float = 0.6
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

        sorted_ids = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k_final]

        out: list[ScoredChunk] = []
        for cid, score in sorted_ids:
            chunk = self.chunks_by_id[cid]
            out.append(ScoredChunk(chunk=chunk, score=score, source="hybrid"))
        return out


# -------------------------------------------------------------------
# Cross-encoder reranker
# -------------------------------------------------------------------


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 10) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs).tolist()
        rescored = []
        for cand, score in zip(candidates, scores):
            rescored.append(ScoredChunk(chunk=cand.chunk, score=float(score), source="reranker"))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]


class NoopReranker:
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 10) -> list[ScoredChunk]:
        candidates = list(candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
