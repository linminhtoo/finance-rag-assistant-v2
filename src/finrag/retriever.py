import json
import pickle
import uuid
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Optional, Protocol, cast

import numpy as np
from loguru import logger
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.sparse import BM25EmbeddingFunction
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from finrag.dataclasses import DocChunk, ScoredChunk
from finrag.llm_clients import LLMClient


class EmbeddingError(RuntimeError):
    pass


class CandidateTextProvider(Protocol):
    def __call__(self, chunk: DocChunk) -> str: ...


# -------------------------------------------------------------------
# Hybrid retriever: Qdrant + BM25
# -------------------------------------------------------------------


class QdrantHybridRetriever:
    """
    Qdrant-backed hybrid retriever (dense + BM25).

    Parameters
    ----------
    llm_client : LLMClient
        LLM client used for dense embeddings.
    storage_path : str
        Qdrant storage path (local or external).
    collection_name : str, optional
        Collection name for vectors and payloads.
    vector_dim : int | None, optional
        Optional dense vector dimension override.
    load_existing : bool, optional
        Whether to load existing collection metadata and BM25 parameters when available.
    load_batch_size : int, optional
        Batch size for loading existing points.
    bm25_path : str | None, optional
        Optional path for BM25 snapshot persistence.
    context_builder : Callable[[DocChunk], str] | None, optional
        Optional callback to build contextual text appended to embeddings.
    index_text_key : str, optional
        Metadata key containing enriched text for embedding (default: "index_text").
    context_metadata_key : str, optional
        Metadata key used to persist contextual text (default: "context").
    """

    def __init__(
        self,
        llm_client: LLMClient,
        storage_path: str,
        collection_name: str = "rag_chunks",
        vector_dim: int | None = None,
        *,
        load_existing: bool = True,
        load_batch_size: int = 512,
        bm25_path: str | None = None,
        context_builder: Callable[[DocChunk], str] | None = None,
        index_text_key: str = "index_text",
        context_metadata_key: str = "context",
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
        self._bm25_path = bm25_path
        self._bm25_dirty = False
        self._context_builder = context_builder
        self._index_text_key = index_text_key
        self._context_metadata_key = context_metadata_key

        if load_existing and self.qdrant.collection_exists(collection_name):
            self._load_existing(batch_size=load_batch_size)
            self._load_bm25()

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
                text, _context = self._text_for_embedding(ch, use_builder=False)
                self._bm25_corpus.append(str(text).split())
                self._bm25_chunk_ids.append(point_id)
                seen += 1
            if offset is None:
                break
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    def existing_chunk_ids(self, chunk_ids: Iterable[str]) -> set[str]:
        """
        Return the subset of `chunk_ids` that already exist in the collection.

        Notes
        -----
        This relies on the in-memory mapping populated when load_existing=True.
        """

        return {chunk_id for chunk_id in chunk_ids if chunk_id in self._qdrant_id_by_chunk_id}

    def text_for_rerank(self, chunk: DocChunk) -> str:
        """
        Text used when reranking retrieved chunks.

        Notes
        -----
        For Qdrant, prefer the already-persisted contextual metadata (if present)
        rather than recomputing context via a builder.
        """

        text, _context = self._text_for_embedding(chunk, use_builder=False)
        return text

    def _load_bm25(self) -> None:
        if not self._bm25_path:
            return
        path = Path(self._bm25_path)
        if not path.exists():
            return
        with path.open("rb") as f:
            payload = pickle.load(f)
        corpus = payload.get("corpus")
        chunk_ids = payload.get("chunk_ids")
        if isinstance(corpus, list) and isinstance(chunk_ids, list) and len(corpus) == len(chunk_ids):
            self._bm25_corpus = corpus
            self._bm25_chunk_ids = chunk_ids
            self._bm25 = BM25Okapi(self._bm25_corpus)
            self._bm25_dirty = False

    def save_bm25(self, path: str | None = None) -> None:
        out_path = Path(path or self._bm25_path or "")
        if not out_path:
            raise ValueError("BM25 path is not set")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"corpus": self._bm25_corpus, "chunk_ids": self._bm25_chunk_ids}
        with out_path.open("wb") as f:
            pickle.dump(payload, f)
        self._bm25_dirty = False

    def rebuild_bm25(self) -> None:
        if not self._bm25_corpus:
            self._bm25 = None
            self._bm25_dirty = False
            return
        self._bm25 = BM25Okapi(self._bm25_corpus)
        self._bm25_dirty = False

    def index(self, chunks: list[DocChunk], *, rebuild_bm25: bool = True) -> None:
        """
        Index document chunks into Qdrant.

        Parameters
        ----------
        chunks : list[DocChunk]
            Chunks to embed and insert.
        rebuild_bm25 : bool, optional
            Whether to rebuild the BM25 corpus after indexing.
        """

        if not chunks:
            return

        texts_with_context: list[str] = []
        contexts: list[str | None] = []
        for chunk in chunks:
            text, context = self._text_for_embedding(chunk, use_builder=True)
            texts_with_context.append(text)
            contexts.append(context)

        try:
            embeddings = self.llm.embed_texts(texts_with_context)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(texts_with_context)} texts for Qdrant indexing into {self.collection_name!r}"
            ) from exc
        dim = int(embeddings.shape[1])
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            self.vector_dim = dim
        elif self.vector_dim is not None and dim != self.vector_dim:
            raise ValueError(f"Embedding dim {dim} != collection dim {self.vector_dim} for {self.collection_name}")

        points = []
        for emb, chunk, context in zip(embeddings, chunks, contexts):
            point_id = self._qdrant_id_for_chunk_id(chunk.id)
            self.chunks_by_id[point_id] = chunk
            self._qdrant_id_by_chunk_id[chunk.id] = point_id
            payload = self._payload_from_chunk(chunk, context)
            points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

        self.qdrant.upsert(collection_name=self.collection_name, points=points, wait=True)

        # BM25
        # TODO: should we add some preprocessing like lowercasing, removing stopwords, etc.?
        # rmbr to add these preprocessing when querying as well
        for text, chunk in zip(texts_with_context, chunks):
            tokens = str(text).split()
            self._bm25_corpus.append(tokens)
            self._bm25_chunk_ids.append(self._qdrant_id_by_chunk_id[chunk.id])
        self._bm25_dirty = True
        if rebuild_bm25:
            self.rebuild_bm25()

    def _semantic_search(self, query: str, top_k: int = 20) -> list[tuple]:
        try:
            q_emb = self.llm.embed_texts([query])[0]
        except Exception as exc:
            logger.warning("Dense query embedding failed (Qdrant); falling back to BM25-only: %r", exc)
            return []
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
        """
        Retrieve chunks using dense + BM25 hybrid scoring.

        Parameters
        ----------
        query : str
            Query text to search.
        top_k_semantic : int, optional
            Candidate count from dense retrieval.
        top_k_bm25 : int, optional
            Candidate count from BM25 retrieval.
        top_k_final : int, optional
            Final result count.
        alpha : float, optional
            Mixing parameter between dense and BM25 scores.

        Returns
        -------
        list[ScoredChunk]
            Retrieved chunks with relevance scores.
        """

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

    def _text_for_embedding(self, chunk: DocChunk, *, use_builder: bool) -> tuple[str, str | None]:
        # TODO: duplicate code with MilvusContextualRetriever._text_for_embedding(),
        # consider refactoring into a shared utility function / base class / MixIn.
        base_text = (chunk.metadata or {}).get(self._index_text_key) or chunk.text
        context = None
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        existing = meta.get(self._context_metadata_key) if meta else None
        if existing:
            context = str(existing)
        if use_builder and self._context_builder is not None:
            built = self._context_builder(chunk).strip()
            if built:
                context = built
        if context:
            return f"{base_text}\n\n{context}", context
        return base_text, None

    def _payload_from_chunk(self, chunk: DocChunk, context: str | None) -> dict[str, Any]:
        payload = chunk.as_payload()
        if context:
            meta = payload.get("metadata")
            if isinstance(meta, dict):
                meta = dict(meta)
            else:
                meta = {}
            meta[self._context_metadata_key] = context
            payload["metadata"] = meta
        return payload


# -------------------------------------------------------------------
# Milvus retriever: dense + optional sparse (BM25/BGE)
# -------------------------------------------------------------------


class DenseEmbedder(Protocol):
    @property
    def dim(self) -> Any: ...

    def __call__(self, texts: list[str]) -> Any: ...


class SparseEmbedder(Protocol):
    def encode_documents(self, documents: list[str]) -> Any: ...

    def encode_queries(self, queries: list[str]) -> Any: ...


class LLMDenseEmbedder:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._dim: int | None = None

    @property
    def dim(self) -> int | None:
        return self._dim

    def __call__(self, texts: list[str]) -> Any:
        try:
            embeddings = self._llm.embed_texts(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed {len(texts)} texts with LLMClient") from exc
        if self._dim is None:
            self._dim = int(embeddings.shape[1])
        return embeddings


def build_milvus_embedding_functions(
    *, llm_client_for_dense: LLMClient, dense_kind: str, sparse_kind: str, use_sparse: bool
) -> tuple[DenseEmbedder, SparseEmbedder | BGEM3EmbeddingFunction | None]:
    """
    Build Milvus dense/sparse embedding functions.

    Parameters
    ----------
    llm_client_for_dense : LLMClient
        LLM client used for dense embeddings when dense_kind="llm".
    dense_kind : str
        Dense embedding backend ("llm" or "bge-m3").
    sparse_kind : str
        Sparse embedding backend ("bm25", "bge-m3", or "none").
    use_sparse : bool
        Whether sparse embeddings are enabled.

    Returns
    -------
    tuple
        (dense_embedding_function, sparse_embedding_function_or_None)
    """
    dense_kind = dense_kind.strip().lower()
    sparse_kind = sparse_kind.strip().lower()

    dense: DenseEmbedder
    sparse: SparseEmbedder | BGEM3EmbeddingFunction | None = None

    if dense_kind == "llm":
        dense = LLMDenseEmbedder(llm_client_for_dense)
    elif dense_kind == "bge-m3":
        if use_sparse and sparse_kind == "bge-m3":
            dense = BGEM3EmbeddingFunction(return_dense=True, return_sparse=True)
            sparse = dense
        else:
            dense = BGEM3EmbeddingFunction(return_dense=True, return_sparse=False)
    else:
        raise ValueError(f"Unknown dense embedding backend: {dense_kind}")

    if use_sparse and sparse is None:
        if sparse_kind == "bm25":
            sparse = BM25EmbeddingFunction()
        elif sparse_kind == "bge-m3":
            sparse = BGEM3EmbeddingFunction(return_dense=False, return_sparse=True)
        elif sparse_kind == "none":
            sparse = None
        else:
            raise ValueError(f"Unknown sparse embedding backend: {sparse_kind}")

    return dense, sparse


class MilvusContextualRetriever:
    """
    Milvus-backed retriever with optional sparse vectors and contextual embeddings.

    Notes
    -----
    - One issue with this design is the storing of sparse vectors in Milvus,
        which makes online ingestion with BM25 or other sparse methods problematic.
        To keep sparse vectors consistent, BM25 should be fitted/loaded and the
        entire index re-built in batch after ingestion.
        An option is to use BGE-M3 sparse embeddings which do not require fitting.
        Lastly, one could disable sparse vectors for online ingestion.

    Parameters
    ----------
    llm_client : LLMClient | None
        LLM client used for dense embeddings when `dense_embedding_function` is not provided.
    uri : str
        Milvus URI or local path (Milvus Lite).
    collection_name : str, optional
        Collection name for vectors and payloads.
    vector_dim : int | None, optional
        Optional dense vector dimension override.
    load_existing : bool, optional
        Whether to load existing collection metadata and BM25 parameters when available.
    use_sparse : bool, optional
        Whether to store/search sparse vectors.
    sparse_embedding_function : SparseEmbedder | BGEM3EmbeddingFunction | None, optional
        Sparse embedding function (BM25EmbeddingFunction or compatible). If None and
        `use_sparse` is True, BM25EmbeddingFunction is used.
    bm25_path : str | None, optional
        Optional path for BM25 parameters (load/save).
    context_builder : Callable[[DocChunk], str] | None, optional
        Optional callback to build contextual text appended to embeddings.
    index_text_key : str, optional
        Metadata key containing enriched text for embedding (default: "index_text").
    context_metadata_key : str, optional
        Metadata key used to persist contextual text (default: "context").
    dense_embedding_function : DenseEmbedder | None, optional
        Dense embedding function. If None, LLMClient embeddings are used.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        uri: str,
        collection_name: str = "rag_chunks",
        vector_dim: int | None = None,
        load_existing: bool = True,
        use_sparse: bool = False,
        sparse_embedding_function: SparseEmbedder | BGEM3EmbeddingFunction | None = None,
        bm25_path: str | None = None,
        context_builder: Callable[[DocChunk], str] | None = None,
        index_text_key: str = "index_text",
        context_metadata_key: str = "context",
        dense_embedding_function: DenseEmbedder | None = None,
    ):
        if dense_embedding_function is None:
            if llm_client is None:
                raise ValueError("llm_client is required when dense_embedding_function is not provided.")
            dense_embedding_function = LLMDenseEmbedder(llm_client)

        self.collection_name = collection_name
        self.client = MilvusClient(uri)

        self.dense_embedding_function = dense_embedding_function
        self.vector_dim = vector_dim
        self.use_sparse = bool(use_sparse)
        self.sparse_embedding_function = sparse_embedding_function
        self._bm25_path = bm25_path
        self._bm25_ready = False

        if self.use_sparse and self.sparse_embedding_function is None:
            if isinstance(dense_embedding_function, BGEM3EmbeddingFunction):
                self.sparse_embedding_function = dense_embedding_function
            else:
                self.sparse_embedding_function = BM25EmbeddingFunction()

        self._context_builder = context_builder
        self._index_text_key = index_text_key
        self._context_metadata_key = context_metadata_key

        if load_existing:
            self._load_existing_collection()
            self._load_bm25()

    @property
    def uses_bm25(self) -> bool:
        return isinstance(self.sparse_embedding_function, BM25EmbeddingFunction)

    def _load_existing_collection(self) -> None:
        if not self.client.has_collection(self.collection_name):
            return
        if self.vector_dim is not None:
            return
        try:
            info = cast(dict[str, Any], self.client.describe_collection(self.collection_name))
        except Exception:
            return
        fields = info.get("fields") or info.get("schema", {}).get("fields")
        if not isinstance(fields, list):
            return
        for field in fields:
            if not isinstance(field, dict):
                continue
            if field.get("name") == "dense_vector":
                params = field.get("params") or {}
                dim = params.get("dim")
                if isinstance(dim, int):
                    self.vector_dim = dim
                break

    def _load_bm25(self) -> None:
        if not (self.use_sparse and self.uses_bm25 and self._bm25_path):
            return
        path = Path(self._bm25_path)
        if not path.exists():
            return
        self.load_bm25(str(path))

    def existing_chunk_ids(self, chunk_ids: Iterable[str], *, batch_size: int = 256) -> set[str]:
        """
        Return the subset of `chunk_ids` that already exist in the collection.
        """

        if not self.client.has_collection(self.collection_name):
            return set()

        ids = [str(cid) for cid in chunk_ids if cid]
        if not ids:
            return set()

        found: set[str] = set()
        bs = max(1, int(batch_size))
        for i in range(0, len(ids), bs):
            batch = ids[i : i + bs]
            try:
                records = self.client.get(collection_name=self.collection_name, ids=batch, output_fields=["chunk_id"])
            except TypeError:
                records = self.client.get(self.collection_name, batch, output_fields=["chunk_id"])
            if not records:
                continue
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                value = rec.get("chunk_id") or rec.get("id")
                if value:
                    found.add(str(value))

        return found

    def _dense_dim(self) -> int:
        dim = getattr(self.dense_embedding_function, "dim", None)
        if isinstance(dim, int):
            return dim
        if isinstance(dim, dict) and isinstance(dim.get("dense"), int):
            return int(dim["dense"])
        raise ValueError("Dense embedding dimension is not available. Provide vector_dim or embed once first.")

    def build_collection(self, *, vector_dim: int | None = None) -> None:
        """
        Create the Milvus collection for dense/sparse vectors and payloads.

        Parameters
        ----------
        vector_dim : int | None, optional
            Dense vector dimension. Defaults to the embedding function's dimension.
        """

        dense_dim = int(vector_dim or self.vector_dim or self._dense_dim())

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
        if self.use_sparse:
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="payload", datatype=DataType.JSON)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="IP")
        if self.use_sparse:
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params, enable_dynamic_field=True
        )
        self.vector_dim = dense_dim

    def fit_bm25(self, corpus: list[str]) -> None:
        """
        Fit BM25 parameters for sparse embeddings.

        Parameters
        ----------
        corpus : list[str]
            Corpus used to fit BM25 statistics.

        Raises
        ------
        RuntimeError
            If sparse embeddings are not configured for BM25.
        """

        if not self.uses_bm25:
            raise RuntimeError("BM25 fitting is only available when using BM25EmbeddingFunction.")
        if self.sparse_embedding_function is None:
            raise RuntimeError("Sparse embedding function is not configured.")
        self.sparse_embedding_function.fit(corpus)  # type: ignore[call-arg]
        self._bm25_ready = True

    def load_bm25(self, path: str) -> None:
        """
        Load BM25 parameters from disk.

        Parameters
        ----------
        path : str
            Path to BM25 parameters saved by `save_bm25`.
        """

        if not self.uses_bm25:
            raise RuntimeError("BM25 loading is only available when using BM25EmbeddingFunction.")
        if self.sparse_embedding_function is None:
            raise RuntimeError("Sparse embedding function is not configured.")
        self.sparse_embedding_function.load(path)  # type: ignore[call-arg]
        self._bm25_ready = True

    def save_bm25(self, path: str | None = None) -> None:
        """
        Persist BM25 parameters to disk.

        Parameters
        ----------
        path : str | None, optional
            Output path. Defaults to the configured bm25_path.
        """

        if not self.uses_bm25:
            raise RuntimeError("BM25 saving is only available when using BM25EmbeddingFunction.")
        out_path = Path(path or self._bm25_path or "")
        if not out_path:
            raise ValueError("BM25 path is not set")
        if self.sparse_embedding_function is None:
            raise RuntimeError("Sparse embedding function is not configured.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.sparse_embedding_function.save(str(out_path))  # type: ignore[call-arg]

    def _sparse_from_documents(self, texts: list[str]) -> Any | None:
        if not self.use_sparse:
            return None
        if self.sparse_embedding_function is None:
            raise RuntimeError("use_sparse=True but sparse_embedding_function is None")
        try:
            if isinstance(self.sparse_embedding_function, BGEM3EmbeddingFunction):
                return self.sparse_embedding_function(texts)["sparse"]
            return self.sparse_embedding_function.encode_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to compute sparse document embeddings for {len(texts)} texts") from exc

    def _sparse_from_queries(self, texts: list[str]) -> Any | None:
        if not self.use_sparse:
            return None
        if self.sparse_embedding_function is None:
            raise RuntimeError("use_sparse=True but sparse_embedding_function is None")
        try:
            if isinstance(self.sparse_embedding_function, BGEM3EmbeddingFunction):
                return self.sparse_embedding_function(texts)["sparse"]
            return self.sparse_embedding_function.encode_queries(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to compute sparse query embeddings for {len(texts)} texts") from exc

    @staticmethod
    def _sparse_row(matrix: Any, idx: int) -> Any:
        try:
            return matrix[[idx]]
        except Exception:
            return matrix[idx]

    def _embed_dense(self, texts: list[str]) -> np.ndarray:
        try:
            embeddings = self.dense_embedding_function(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to compute dense embeddings for {len(texts)} texts") from exc
        if isinstance(self.dense_embedding_function, BGEM3EmbeddingFunction):
            dense = embeddings.get("dense") if isinstance(embeddings, dict) else None
            if dense is None:
                raise RuntimeError("BGEM3EmbeddingFunction did not return dense embeddings.")
            return np.asarray(dense, dtype=np.float32)
        if isinstance(embeddings, np.ndarray):
            return embeddings
        return np.asarray(embeddings, dtype=np.float32)

    def _text_for_embedding(self, chunk: DocChunk) -> tuple[str, str | None]:
        base_text = (chunk.metadata or {}).get(self._index_text_key) or chunk.text
        context = None
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        existing = meta.get(self._context_metadata_key) if meta else None
        if existing:
            context = str(existing)
        if self._context_builder is not None:
            built = self._context_builder(chunk).strip()
            if built:
                context = built
        if context:
            return f"{base_text}\n\nContext: {context}", context
        return base_text, None

    def text_for_rerank(self, chunk: DocChunk) -> str:
        """
        Text used when reranking retrieved chunks.

        This intentionally reuses the same logic as `_text_for_embedding()` so
        reranking sees the full enriched chunk representation (e.g. index text
        and contextual metadata).
        """

        text, _context = self._text_for_embedding(chunk)
        return text

    def _payload_from_chunk(self, chunk: DocChunk, context: str | None) -> dict[str, Any]:
        payload = chunk.as_payload()
        if context:
            meta = payload.get("metadata")
            if isinstance(meta, dict):
                meta = dict(meta)
            else:
                meta = {}
            meta[self._context_metadata_key] = context
            payload["metadata"] = meta
        return payload

    def index(self, chunks: list[DocChunk], *, rebuild_bm25: bool = True) -> None:
        """
        Index document chunks into Milvus.

        Parameters
        ----------
        chunks : list[DocChunk]
            Chunks to embed and insert.
        rebuild_bm25 : bool, optional
            Reserved for API parity. When using BM25 sparse embeddings,
            call `fit_bm25` or `load_bm25` before indexing.

        Raises
        ------
        RuntimeError
            If BM25 sparse embeddings are enabled but not fitted/loaded.
        """

        if not chunks:
            return
        if self.use_sparse and self.uses_bm25 and not self._bm25_ready:
            raise RuntimeError("BM25 sparse embeddings are not ready. Call fit_bm25() or load_bm25() first.")

        texts_with_context: list[str] = []
        contexts: list[str | None] = []
        for chunk in chunks:
            text, context = self._text_for_embedding(chunk)
            texts_with_context.append(text)
            contexts.append(context)

        dense_vectors = self._embed_dense(texts_with_context)
        if dense_vectors.shape[0] != len(chunks):
            raise RuntimeError("Dense embedding count does not match chunk count.")

        if self.vector_dim is not None and int(dense_vectors.shape[1]) != int(self.vector_dim):
            raise ValueError(
                f"Embedding dim {dense_vectors.shape[1]} != collection dim {self.vector_dim} for {self.collection_name}"
            )
        if not self.client.has_collection(self.collection_name):
            self.build_collection(vector_dim=int(self.vector_dim or dense_vectors.shape[1]))

        sparse_vectors = self._sparse_from_documents(texts_with_context) if self.use_sparse else None

        data: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            payload = self._payload_from_chunk(chunk, contexts[idx])
            row = {"chunk_id": chunk.id, "dense_vector": dense_vectors[idx].tolist(), "payload": payload}
            if self.use_sparse:
                if sparse_vectors is None:
                    raise RuntimeError("use_sparse=True but sparse_vectors is None")
                row["sparse_vector"] = self._sparse_row(sparse_vectors, idx)
            data.append(row)

        if hasattr(self.client, "upsert"):
            self.client.upsert(collection_name=self.collection_name, data=data)
        else:
            self.client.insert(collection_name=self.collection_name, data=data)

    def retrieve_hybrid(
        self, query: str, top_k_semantic: int = 20, top_k_bm25: int = 20, top_k_final: int = 20, alpha: float = 0.6
    ) -> list[ScoredChunk]:
        """
        Retrieve chunks using dense-only or hybrid (dense+sparse) search.

        Parameters
        ----------
        query : str
            Query text to search.
        top_k_semantic : int, optional
            Candidate count from dense retrieval (hybrid mode).
        top_k_bm25 : int, optional
            Candidate count from sparse retrieval (hybrid mode).
        top_k_final : int, optional
            Final result count.
        alpha : float, optional
            Mixing parameter (unused when Milvus hybrid search uses RRFRanker).

        Returns
        -------
        list[ScoredChunk]
            Retrieved chunks with relevance scores.
        """

        dense_vec: np.ndarray | None
        try:
            dense_vec = self._embed_dense([query])[0]
        except Exception as exc:
            dense_vec = None
            logger.warning("Dense query embedding failed (Milvus); attempting sparse-only fallback: %r", exc)
        output_fields = ["payload"]

        hits: list[dict[str, Any]]
        if not self.client.has_collection(self.collection_name):
            return []

        sparse_vecs = None
        if self.use_sparse:
            try:
                sparse_vecs = self._sparse_from_queries([query])
            except Exception as exc:
                logger.warning("Sparse query embedding failed (Milvus); falling back to dense-only: %r", exc)
                sparse_vecs = None
        if sparse_vecs is None and dense_vec is None:
            raise RuntimeError("Both dense and sparse query embeddings failed; cannot search.")

        if self.use_sparse and sparse_vecs is not None and dense_vec is not None:
            dense_req = AnnSearchRequest(
                data=[dense_vec.tolist()],
                anns_field="dense_vector",
                param={"metric_type": "IP"},
                limit=max(top_k_semantic, top_k_final),
            )
            sparse_req = AnnSearchRequest(
                data=[self._sparse_row(sparse_vecs, 0)],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=max(top_k_bm25, top_k_final),
            )
            results = self.client.hybrid_search(
                self.collection_name, [dense_req, sparse_req], RRFRanker(), top_k_final, output_fields=output_fields
            )
            hits = results[0] if results else []
        elif self.use_sparse and sparse_vecs is not None and dense_vec is None:
            results = self.client.search(
                self.collection_name,
                data=[self._sparse_row(sparse_vecs, 0)],
                anns_field="sparse_vector",
                limit=top_k_final,
                output_fields=output_fields,
            )
            hits = results[0] if results else []
        elif dense_vec is not None:
            results = self.client.search(
                self.collection_name,
                data=[dense_vec.tolist()],
                anns_field="dense_vector",
                limit=top_k_final,
                output_fields=output_fields,
            )
            hits = results[0] if results else []
        else:
            return []

        out: list[ScoredChunk] = []
        for hit in hits:
            entity = hit.get("entity") if isinstance(hit, Mapping) else None
            payload = None
            if isinstance(entity, dict):
                payload = entity.get("payload")
            if payload is None and isinstance(hit, Mapping):
                payload = hit.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if not isinstance(payload, dict):
                payload = {}

            hit_id = hit.get("id") if isinstance(hit, Mapping) else getattr(hit, "id", "")
            chunk = DocChunk(
                id=str(payload.get("chunk_id") or hit_id or ""),
                doc_id=str(payload.get("doc_id") or ""),
                text=str(payload.get("text") or ""),
                page_no=payload.get("page_no"),
                headings=list(payload.get("headings") or []),
                source=str(payload.get("source") or ""),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            )
            score = hit.get("score") if isinstance(hit, Mapping) else getattr(hit, "score", None)
            if score is None:
                score = hit.get("distance") if isinstance(hit, Mapping) else getattr(hit, "distance", None)
            out.append(ScoredChunk(chunk=chunk, score=float(score or 0.0), source="hybrid"))

        return out


# -------------------------------------------------------------------
# Cross-encoder reranker
# -------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Cross-encoder reranker using SentenceTransformers.

    Parameters
    ----------
    model_name : str, optional
        Pretrained cross-encoder model name.
        Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
        Users can also experiment with "BAAI/bge-reranker-v2-gemma".
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        candidate_text_provider: CandidateTextProvider | None = None,
    ):
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        self._candidate_text_provider = candidate_text_provider

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 10,
        *,
        candidate_text_provider: CandidateTextProvider | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        provider = candidate_text_provider or self._candidate_text_provider
        if provider is None:
            pairs = [(query, c.chunk.text) for c in candidates]
        else:
            pairs = [(query, provider(c.chunk)) for c in candidates]
        scores = self.model.predict(pairs).tolist()
        rescored = []
        for cand, score in zip(candidates, scores):
            rescored.append(ScoredChunk(chunk=cand.chunk, score=float(score), source="reranker"))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]


class NoopReranker:
    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 10,
        *,
        candidate_text_provider: CandidateTextProvider | None = None,
    ) -> list[ScoredChunk]:
        candidates = list(candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
