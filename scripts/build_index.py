"""
Build a hybrid index (Qdrant vectors + in-memory BM25) from exported chunks.

Expected input is the output directory produced by `scripts/ingest.py`, which
contains:
  - doc_index.jsonl
  - chunks/ (per-document JSONL chunk files)

This script embeds and upserts the chunks into a local Qdrant store.
BM25 is rebuilt at runtime from stored Qdrant payloads in `HybridRetriever`.
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
from tqdm import tqdm

from finrag.dataclasses import DocChunk
from finrag.llm_clients import get_llm_client
from finrag.retriever import HybridRetriever


@dataclass
class Args:
    ingest_output_dir: str
    storage_path: str
    collection_name: str
    llm_provider: str | None
    embedding_model: str
    cache_dir: str | None
    bm25_path: str | None
    max_docs: int | None
    overwrite_collection: bool
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Build a hybrid index from chunk exports (Qdrant + BM25).")
    parser.add_argument(
        "--ingest-output-dir",
        required=True,
        help="Directory produced by `scripts/ingest.py` (must contain doc_index.jsonl and chunks/).",
    )
    parser.add_argument(
        "--storage-path", default=None, help="Qdrant storage path (defaults to env QDRANT_STORAGE_PATH if set)."
    )
    parser.add_argument("--collection-name", default="rag_chunks", help="Qdrant collection name.")
    parser.add_argument(
        "--llm-provider",
        default=None,
        help=(
            "Embedding provider (defaults to env LLM_PROVIDER). "
            "Choices are: 'openai', 'mistral', 'fastembed'. "
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Name of embedding model. Used by all llm providers.",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Directory to use for model caches/temp files (keeps writes inside the repo)."
    )
    parser.add_argument(
        "--bm25-path", default=None, help="Where to persist BM25 data (defaults to <ingest_output_dir>/bm25.pkl)."
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on number of documents to index.")
    parser.add_argument(
        "--overwrite-collection",
        action="store_true",
        help="Delete and recreate the collection before indexing (destructive).",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Embed/upsert batch size (within a document).")

    args = parser.parse_args()

    storage_path = args.storage_path or os.environ.get("QDRANT_STORAGE_PATH")
    if not storage_path:
        raise SystemExit("Missing --storage-path and QDRANT_STORAGE_PATH is not set.")

    return Args(
        ingest_output_dir=args.ingest_output_dir,
        storage_path=storage_path,
        collection_name=args.collection_name,
        llm_provider=args.llm_provider,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
        bm25_path=args.bm25_path,
        max_docs=args.max_docs,
        overwrite_collection=bool(args.overwrite_collection),
        batch_size=args.batch_size,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass
    return str(obj)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _chunk_from_dict(d: dict[str, Any]) -> DocChunk:
    return DocChunk(
        id=str(d["id"]),
        doc_id=str(d["doc_id"]),
        text=str(d["text"]),
        page_no=d.get("page_no"),
        headings=list(d.get("headings") or []),
        source=str(d.get("source") or ""),
        metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else None,
    )


def _batched(items: list[DocChunk], batch_size: int) -> Iterable[list[DocChunk]]:
    if batch_size <= 0:
        yield items
        return
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> int:
    args = parse_args()
    ingest_out = Path(args.ingest_output_dir).expanduser().resolve()
    doc_index_path = ingest_out / "doc_index.jsonl"
    bm25_path = Path(args.bm25_path).expanduser().resolve() if args.bm25_path else (ingest_out / "bm25.pkl").resolve()

    if not doc_index_path.exists():
        raise SystemExit(f"Missing required file: {doc_index_path}")

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else (ingest_out / ".cache").resolve()
    tmp_dir = (ingest_out / ".tmp").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Keep model downloads/cache/temp writes inside the project tree.
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "huggingface" / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"build_index_{ts}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="DEBUG")
    logger.info(f"Logging to: {log_path}")

    llm_kwargs: dict[str, Any] = {}
    if (args.llm_provider or "").strip().lower() in {"fastembed", "local"}:
        llm_kwargs["embed_model"] = args.embedding_model
    llm = get_llm_client(provider=args.llm_provider, **llm_kwargs)

    retriever = HybridRetriever(
        llm,
        storage_path=args.storage_path,
        collection_name=args.collection_name,
        load_existing=not args.overwrite_collection,
        bm25_path=str(bm25_path),
    )

    if args.overwrite_collection and retriever.qdrant.collection_exists(args.collection_name):
        logger.warning(f"Deleting collection: {args.collection_name}")
        retriever.qdrant.delete_collection(args.collection_name)

    docs = list(_iter_jsonl(doc_index_path))
    if args.max_docs is not None:
        docs = docs[: max(0, args.max_docs)]
    logger.info(f"Documents to index: {len(docs)}")

    started_at = time.time()
    total_docs = 0
    total_chunks = 0
    had_error = False

    def _handle_signal(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        for doc in tqdm(docs, desc="indexing docs"):
            chunks_path = Path(str(doc.get("chunks_path") or "")).expanduser()
            if not chunks_path.is_absolute():
                chunks_path = (ingest_out / chunks_path).resolve()
            if not chunks_path.exists():
                logger.warning(f"Missing chunk file, skipping: {chunks_path}")
                continue

            doc_chunks = [_chunk_from_dict(d) for d in _iter_jsonl(chunks_path)]
            if not doc_chunks:
                continue

            for batch in _batched(doc_chunks, batch_size=args.batch_size):
                # NOTE: avoid repeatedly rebuilding BM25 until all docs are indexed
                # TODO: look into alternative sparse retrieval indices like OpenSearch
                # Milvus ???
                # In this tutorial, we will demonstrate how to conduct hybrid search with Milvus and BGE-M3 model.
                # BGE-M3 model can convert text into dense and sparse vectors. 
                # Milvus supports storing both types of vectors in one collection, allowing for hybrid search 
                # that enhances the result relevance.
                retriever.index(batch, rebuild_bm25=False)

            total_docs += 1
            total_chunks += len(doc_chunks)
    except BaseException as e:  # noqa: BLE001 - want to catch signals/interrupts too
        had_error = True
        logger.exception(f"Indexing interrupted: {e}")
    finally:
        if total_chunks > 0:
            try:
                retriever.rebuild_bm25()
                retriever.save_bm25(str(bm25_path))
                logger.info(f"Saved BM25 snapshot to: {bm25_path}")
            except Exception as e:  # noqa: BLE001 - best-effort snapshot
                logger.exception(f"Failed to save BM25 snapshot: {e}")

    out = {
        "args": args.to_dict(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - started_at, 3),
        "ingest_output_dir": str(ingest_out),
        "qdrant_storage_path": args.storage_path,
        "collection_name": args.collection_name,
        "indexed_docs": total_docs,
        "indexed_chunks": total_chunks,
    }
    run_info_path = ingest_out / "build_index_run_info.json"
    run_info_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    if had_error:
        logger.warning(f"Indexing stopped early. indexed_docs={total_docs} indexed_chunks={total_chunks}")
    else:
        logger.success(f"Done. indexed_docs={total_docs} indexed_chunks={total_chunks}")
    logger.success(f"Wrote: {run_info_path}")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
