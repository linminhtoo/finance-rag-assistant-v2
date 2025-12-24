"""
Adapted from:
https://milvus.io/docs/contextual_retrieval_with_milvus.md
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    RRFRanker,
)
# from pymilvus.model.dense import OpenAIEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction, CrossEncoderRerankFunction
from tqdm import tqdm


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


MilvusHit = dict[str, Any]
MilvusSearchResults = list[list[MilvusHit]]


class RerankResultLike(Protocol):
    index: int


class Reranker(Protocol):
    def __call__(self, query: str, documents: list[str], top_k: int = 5) -> list[RerankResultLike]: ...


class ChunkRecord(TypedDict):
    chunk_id: str
    original_index: int
    content: str


class DocumentRecord(TypedDict, total=False):
    doc_id: str
    original_uuid: str
    content: str
    chunks: list[ChunkRecord]


class EvalResults(TypedDict):
    pass_at_n: float
    average_score: float
    total_queries: int


class MilvusContextualRetriever:
    def __init__(
        self,
        *,
        uri: str = "./data/milvus.db",
        collection_name: str = "example",
        # NOTE: typehinting as BaseEmbeddingFunction results in a lot of pylance errors
        # bcos pymilvus did not typehint return value of __call__
        # we can bypass it by using Protocol
        embedding_function: BGEM3EmbeddingFunction,
        use_sparse: bool = True,
        use_contextualize_embedding: bool = False,
        openai_client: OpenAI | None = None,
        context_model: str = "",
        context_max_tokens: int = 1000,
        use_reranker: bool = False,
        rerank_function: Reranker | None = None,
        rerank_top_k: int = 5,
    ):
        self.collection_name = collection_name

        # For Milvus-lite, uri is a local path like "./milvus.db"
        # For Milvus standalone service, uri is like "http://localhost:19530"
        # For Zilliz Clond, please set `uri` and `token`, which correspond to the [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details) in Zilliz Cloud.
        self.client = MilvusClient(uri)

        self.embedding_function = embedding_function

        self.use_sparse = use_sparse

        self.use_contextualize_embedding = use_contextualize_embedding
        self.openai_client = openai_client
        self.context_model = context_model
        self.context_max_tokens = context_max_tokens

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function
        self.rerank_top_k = rerank_top_k

        if self.use_contextualize_embedding and self.openai_client is None:
            raise ValueError("OpenAI client must be provided when use_contextualize_embedding=True")
        if self.use_reranker and self.rerank_function is None:
            raise ValueError("rerank_function must be provided when use_reranker=True")

    def build_collection(self):
        dense_dim = self.embedding_function.dim.get("dense")
        if not isinstance(dense_dim, int):
            raise ValueError(f"Unexpected embedding dim for dense vectors: {self.embedding_function.dim!r}")

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dense_dim,
        )
        if self.use_sparse:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )
        if self.use_sparse:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk: str, metadata: dict[str, Any]) -> None:
        embeddings = self.embedding_function([chunk])
        dense_vec = embeddings["dense"][0]
        if self.use_sparse:
            sparse_vec = embeddings["sparse"][[0]]
            self.client.insert(
                collection_name=self.collection_name,
                data={
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    **metadata,
                },
            )
        else:
            self.client.insert(
                collection_name=self.collection_name,
                data={"dense_vector": dense_vec, **metadata},
            )

    def insert_contextualized_data(self, doc: str, chunk: str, metadata: dict[str, Any]) -> None:
        contextualized_text, _usage = self.situate_context(doc, chunk)
        if contextualized_text:
            metadata["context"] = contextualized_text
            text_to_embed = f"{chunk}\n\n{contextualized_text}"
        else:
            text_to_embed = chunk
        embeddings = self.embedding_function([text_to_embed])
        dense_vec = embeddings["dense"][0]
        if self.use_sparse:
            sparse_vec = embeddings["sparse"][[0]]
            self.client.insert(
                collection_name=self.collection_name,
                data={
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    **metadata,
                },
            )
        else:
            self.client.insert(
                collection_name=self.collection_name,
                data={"dense_vector": dense_vec, **metadata},
            )

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        if self.openai_client is None:
            logger.warning("OpenAI client not configured; skipping contextualization.")
            return "", None

        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.context_model,
                max_tokens=self.context_max_tokens,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            },
                            {
                                "type": "text",
                                "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                            },
                        ],
                    },
                ],
            )
            content = response.choices[0].message.content
            if content is None:
                logger.warning("OpenAI response message content is None; skipping contextualization.")
                return "", response.usage
            return content, response.usage
        except Exception as e:
            logger.warning(f"Contextualization call failed; falling back to raw chunk. Error: {e!r}")
            return "", None

    def search(self, query: str, k: int = 20) -> MilvusSearchResults:
        embeddings = self.embedding_function([query])
        dense_vec = embeddings["dense"][0]
        if self.use_sparse:
            sparse_vec = embeddings["sparse"][[0]]

        req_list = []
        if self.use_reranker:
            # When using reranker, we retrieve more candidates to allow reranking to be effective
            k = k * 10

        if self.use_sparse:
            req_list = []
            dense_search_param = {
                "data": [dense_vec],
                "anns_field": "dense_vector",
                "param": {"metric_type": "IP"},
                "limit": k * 2,
            }
            dense_req = AnnSearchRequest(**dense_search_param)
            req_list.append(dense_req)

            sparse_search_param = {
                "data": [sparse_vec],
                "anns_field": "sparse_vector",
                "param": {"metric_type": "IP"},
                "limit": k * 2,
            }
            sparse_req = AnnSearchRequest(**sparse_search_param)

            req_list.append(sparse_req)

            docs: MilvusSearchResults = self.client.hybrid_search(
                self.collection_name,
                req_list,
                RRFRanker(),
                k,
                output_fields=[
                    "content",
                    "original_uuid",
                    "doc_id",
                    "chunk_id",
                    "original_index",
                    "context",
                ],
            )
        else:
            docs = self.client.search(
                self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=[
                    "content",
                    "original_uuid",
                    "doc_id",
                    "chunk_id",
                    "original_index",
                    "context",
                ],
            )
        
        if self.use_reranker:
            if self.rerank_function is None:
                raise RuntimeError("use_reranker=True but rerank_function is None")

            hits = docs[0]
            candidate_count = min(len(hits), k)
            candidate_texts: list[str] = []
            for i in range(candidate_count):
                entity = hits[i].get("entity") if isinstance(hits[i], dict) else None
                content = ""
                if isinstance(entity, dict):
                    content = str(entity.get("content") or "")
                    if self.use_contextualize_embedding and entity.get("context"):
                        content = f"{content}\n\n{entity.get('context')}"
                candidate_texts.append(content)

            reranked = self.rerank_function(query, candidate_texts, top_k=self.rerank_top_k)
            reranked_docs = [hits[r.index] for r in reranked if 0 <= r.index < len(hits)]
            docs[0] = reranked_docs

        return docs


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    retrieval_function: Callable[[str, Any, int], MilvusSearchResults],
    db: Any,
    k: int = 20,
) -> EvalResults:
    total_score = 0
    total_queries = len(queries)
    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = str(query_item["query"])
        golden_chunk_uuids = query_item["golden_chunk_uuids"]

        # Find all golden chunk contents
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (
                    doc
                    for doc in query_item["golden_documents"]
                    if doc["uuid"] == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                logger.warning(f"Golden document not found for UUID {doc_uuid}")
                continue

            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                logger.warning(
                    f"Golden chunk not found for index {chunk_index} in document {doc_uuid}"
                )
                continue

            golden_contents.append(golden_chunk["content"].strip())

        if not golden_contents:
            logger.warning(f"No golden contents found for query: {query}")
            continue

        retrieved_docs = retrieval_function(query, db, k)

        # Count how many golden chunks are in the top k retrieved documents
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[0][:k]:
                retrieved_content = doc["entity"]["content"].strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break

        query_score = chunks_found / len(golden_contents)
        total_score += query_score

    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries,
    }


def retrieve_base(query: str, db: Any, k: int = 20) -> MilvusSearchResults:
    return db.search(query, k=k)


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def evaluate_db(db: Any, original_jsonl_path: str, k: int) -> None:
    # Load the original JSONL data for queries and ground truth
    original_data = load_jsonl(original_jsonl_path)

    # Evaluate retrieval
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    logger.info(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    logger.info(f"Total Score: {results['average_score']}")
    logger.info(f"Total queries: {results['total_queries']}")


def _env_default(key: str) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _openai_config_from_env() -> tuple[str | None, str | None]:
    api_key = _env_default("OPENAI_API_KEY")
    base_url = _env_default("OPENAI_BASE_URL")
    return api_key, base_url


def _build_reranker(kind: str, model_name: str) -> Reranker:
    kind_norm = kind.strip().lower()
    if kind_norm in {"bge", "bgererank", "bge-reranker"}:
        return BGERerankFunction(model_name=model_name)  # type: ignore
    if kind_norm in {"cross", "cross-encoder", "crossencoder"}:
        return CrossEncoderRerankFunction(model_name=model_name)  # type: ignore
    raise ValueError(f"Unknown reranker type: {kind}")


@dataclass
class Args:
    input_chunks_path: str
    evaluation_set_path: str
    milvus_uri: str
    collection_name: str
    k: int
    recreate_collection: bool
    use_sparse: bool
    use_contextualize_embedding: bool
    context_model: str
    context_max_tokens: int
    use_reranker: bool
    reranker_type: str
    reranker_model_name: str
    rerank_top_k: int
    openai_api_key: str | None
    openai_base_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Milvus hybrid search example (BGE-M3 dense+sparse + optional rerank).")
    parser.add_argument(
        "--input-chunks-path",
        default="codebase_chunks.json",
        help="Path to chunk dataset JSON (e.g. codebase_chunks.json).",
    )
    parser.add_argument(
        "--evaluation-set-path",
        default="evaluation_set.jsonl",
        help="Path to evaluation set JSONL (queries + golden chunks).",
    )
    parser.add_argument("--milvus-uri", default="hybrid.db", help="Milvus URI/path (Milvus Lite uses a local file).")
    parser.add_argument("--collection-name", default="hybrid", help="Milvus collection name.")
    parser.add_argument("--k", type=int, default=20, help="Top-k to return for retrieval.")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the collection before inserting (destructive).",
    )

    sparse_group = parser.add_mutually_exclusive_group()
    sparse_group.add_argument("--use-sparse", dest="use_sparse", action="store_true", help="Enable sparse vectors.")
    sparse_group.add_argument("--no-sparse", dest="use_sparse", action="store_false", help="Disable sparse vectors.")
    parser.set_defaults(use_sparse=True)

    ctx_group = parser.add_mutually_exclusive_group()
    ctx_group.add_argument(
        "--contextualize-embedding",
        dest="use_contextualize_embedding",
        action="store_true",
        help="Generate a context string for each chunk using an OpenAI-compatible API and embed chunk+context.",
    )
    ctx_group.add_argument(
        "--no-contextualize-embedding",
        dest="use_contextualize_embedding",
        action="store_false",
        help="Embed chunks as-is (no OpenAI calls).",
    )
    parser.set_defaults(use_contextualize_embedding=False)
    parser.add_argument(
        "--context-model",
        default="claude-3-haiku-20240307",
        help="Name of model to be used for contextualization, accessed via OpenAI-compatible API.",
    )
    parser.add_argument("--context-max-tokens", type=int, default=1000, help="Max tokens for contextualization calls.")

    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--use-reranker", dest="use_reranker", action="store_true", help="Enable reranking.")
    rerank_group.add_argument("--no-reranker", dest="use_reranker", action="store_false", help="Disable reranking.")
    parser.set_defaults(use_reranker=True)

    parser.add_argument(
        "--reranker-type",
        choices=["bge", "cross-encoder"],
        default="cross-encoder",
        help="Which reranker to use.",
    )
    parser.add_argument(
        "--reranker-model-name",
        default=None,
        help="Model name passed to the chosen reranker.",
    )
    parser.add_argument("--rerank-top-k", type=int, default=5, help="How many reranked hits to keep.")

    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Override OpenAI API key (otherwise loads from .env `api_key` or env OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Override OpenAI base URL (otherwise loads from .env `base_url` or env OPENAI_BASE_URL).",
    )

    ns = parser.parse_args()
    api_key_env, base_url_env = _openai_config_from_env()

    return Args(
        input_chunks_path=str(ns.input_chunks_path),
        evaluation_set_path=str(ns.evaluation_set_path),
        milvus_uri=str(ns.milvus_uri),
        collection_name=str(ns.collection_name),
        k=int(ns.k),
        recreate_collection=bool(ns.recreate_collection),
        use_sparse=bool(ns.use_sparse),
        use_contextualize_embedding=bool(ns.use_contextualize_embedding),
        context_model=str(ns.context_model),
        context_max_tokens=int(ns.context_max_tokens),
        use_reranker=bool(ns.use_reranker),
        reranker_type=str(ns.reranker_type),
        reranker_model_name=(
            str(ns.reranker_model_name)
            if ns.reranker_model_name
            else ("BAAI/bge-reranker-v2-m3" if str(ns.reranker_type) == "bge" else "cross-encoder/ms-marco-MiniLM-L6-v2")
        ),
        rerank_top_k=int(ns.rerank_top_k),
        openai_api_key=str(ns.openai_api_key).strip() if ns.openai_api_key else api_key_env,
        openai_base_url=str(ns.openai_base_url).strip() if ns.openai_base_url else base_url_env,
    )


def _build_openai_client(*, api_key: str | None, base_url: str | None) -> OpenAI | None:
    if not api_key and not base_url:
        return None
    if not api_key:
        raise SystemExit("Missing OpenAI API key: set `.env` `api_key` or pass `--openai-api-key`.")
    if not base_url:
        raise SystemExit("Missing OpenAI base URL: set `.env` `base_url` or pass `--openai-base-url`.")
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": 1,
        "timeout": 120,
    }
    return OpenAI(**kwargs)


def main() -> int:
    args = parse_args()
    logger.info(f"Args: {args.to_dict()}")

    embedding_function = BGEM3EmbeddingFunction()
    reranker: Reranker | None = None
    if args.use_reranker:
        reranker = _build_reranker(args.reranker_type, args.reranker_model_name)

    openai_client = _build_openai_client(api_key=args.openai_api_key, base_url=args.openai_base_url)
    if args.use_contextualize_embedding and openai_client is None:
        raise SystemExit(
            "Contextualized embeddings require OpenAI config: set `.env` `api_key`/`base_url` or pass CLI overrides."
        )

    retriever = MilvusContextualRetriever(
        uri=args.milvus_uri,
        collection_name=args.collection_name,
        embedding_function=embedding_function,
        use_sparse=args.use_sparse,
        use_contextualize_embedding=args.use_contextualize_embedding,
        openai_client=openai_client,
        context_model=args.context_model,
        context_max_tokens=args.context_max_tokens,
        use_reranker=args.use_reranker,
        rerank_function=reranker,
        rerank_top_k=args.rerank_top_k,
    )

    if args.recreate_collection and retriever.client.has_collection(args.collection_name):
        logger.warning(f"Dropping existing collection: {args.collection_name}")
        retriever.client.drop_collection(args.collection_name)

    if not retriever.client.has_collection(args.collection_name):
        retriever.build_collection()

    input_path = Path(args.input_chunks_path).expanduser()
    with input_path.open("r", encoding="utf-8") as f:
        dataset: list[DocumentRecord] = json.load(f)

    t_start = time.time()
    for doc in dataset:
        doc_id = str(doc.get("doc_id") or "")
        original_uuid = str(doc.get("original_uuid") or "")
        doc_content = str(doc.get("content") or "")
        chunks = doc.get("chunks") or []

        for chunk in chunks:
            chunk_content = str(chunk.get("content") or "")
            metadata: dict[str, Any] = {
                "doc_id": doc_id,
                "original_uuid": original_uuid,
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "original_index": int(chunk.get("original_index") or 0),
                "content": chunk_content,
            }
            if args.use_contextualize_embedding and doc_content:
                # TODO: if doc_content is too large (which will happen for our private dataset)
                # we need to find a way to truncate it intelligently
                retriever.insert_contextualized_data(doc_content, chunk_content, metadata)
            else:
                retriever.insert_data(chunk_content, metadata)

    logger.info(f"Data insertion completed in {time.time() - t_start:.2f} seconds.")

    t_eval_start = time.time()
    eval_k = min(args.k, args.rerank_top_k) if args.use_reranker else args.k
    # example eval set can be downloaded from github
    # wget https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl
    evaluate_db(retriever, args.evaluation_set_path, k=eval_k)
    logger.info(f"Evaluation completed in {time.time() - t_eval_start:.2f} seconds.")
    logger.info(f"Total time taken: {time.time() - t_start:.2f} seconds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
