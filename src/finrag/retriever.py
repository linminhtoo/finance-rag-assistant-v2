from typing import Optional

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

    def index(self, chunks: list[DocChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.llm.embed_texts(texts)
        dim = int(embeddings.shape[1])
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self.vector_dim = dim
        elif self.vector_dim is not None and dim != self.vector_dim:
            raise ValueError(f"Embedding dim {dim} != collection dim {self.vector_dim} for {self.collection_name}")

        points = []
        for emb, chunk in zip(embeddings, chunks):
            self.chunks_by_id[chunk.id] = chunk
            points.append(PointStruct(id=chunk.id, vector=emb.tolist(), payload=chunk.as_payload()))

        self.qdrant.upsert(collection_name=self.collection_name, points=points, wait=True)

        # BM25
        for chunk in chunks:
            tokens = chunk.text.split()
            self._bm25_corpus.append(tokens)
            self._bm25_chunk_ids.append(chunk.id)
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
