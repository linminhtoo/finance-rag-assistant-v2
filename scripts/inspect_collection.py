"""
Inspect stored chunks (payload/context) in a Qdrant or Milvus collection.

Examples
--------
Milvus (Milvus Lite file):
  python scripts/inspect_collection.py --backend milvus --milvus-uri ./data/milvus.db --collection-name rag_chunks --limit 5
  python scripts/inspect_collection.py --backend milvus --milvus-uri ./data/milvus.db --collection-name rag_chunks --chunk-id "<chunk_id>"

Qdrant (local storage path):
  python scripts/inspect_collection.py --backend qdrant --qdrant-storage-path ./data/qdrant --collection-name rag_chunks --limit 5
  python scripts/inspect_collection.py --backend qdrant --qdrant-storage-path ./data/qdrant --collection-name rag_chunks --chunk-id "<chunk_id>"
"""

import argparse
import json
import uuid
from typing import Any, cast


def _maybe_truncate(text: str, *, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _extract_context(payload: dict[str, Any], *, context_metadata_key: str) -> str | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = None
    if metadata is not None:
        value = metadata.get(context_metadata_key)
        if value is not None:
            return str(value)
    value = payload.get(context_metadata_key)
    if value is not None:
        return str(value)
    return None


def _print_record(record: dict[str, Any], *, context_metadata_key: str, max_chars: int | None, as_json: bool) -> None:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata = cast(dict[str, Any], metadata)
    out = {
        "chunk_id": record.get("chunk_id") or payload.get("chunk_id"),
        "doc_id": payload.get("doc_id"),
        "source": payload.get("source"),
        "page_no": payload.get("page_no"),
        "headings": payload.get("headings"),
        "context": _maybe_truncate(
            _extract_context(payload, context_metadata_key=context_metadata_key) or "", max_chars=max_chars
        ),
        "text": _maybe_truncate(payload.get("text") or "", max_chars=max_chars),
        "index_text": _maybe_truncate(
            (metadata.get("index_text") or payload.get("index_text") or "not found"), max_chars=max_chars
        ),
    }
    if as_json:
        print(json.dumps(out, ensure_ascii=False))
        return
    print(f"chunk_id: {out['chunk_id']}\n")
    print(f"context: {out['context']}\n")
    print(f"text: {out['text']}\n")
    print(f"index_text: {out['index_text']}\n")
    print("---")


def inspect_milvus(
    *,
    milvus_uri: str,
    collection_name: str,
    chunk_id: str | None,
    limit: int,
    context_metadata_key: str,
    max_chars: int | None,
    as_json: bool,
) -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(milvus_uri)

    if chunk_id:
        try:
            records = client.get(collection_name=collection_name, ids=[chunk_id], output_fields=["chunk_id", "payload"])
        except TypeError:
            records = client.get(collection_name, [chunk_id], output_fields=["chunk_id", "payload"])
    else:
        flt = 'chunk_id != ""'
        try:
            records = client.query(
                collection_name=collection_name, filter=flt, limit=limit, output_fields=["chunk_id", "payload"]
            )
        except TypeError:
            records = client.query(collection_name, flt, limit=limit, output_fields=["chunk_id", "payload"])

    if not records:
        return
    for rec in records:
        if isinstance(rec, dict):
            _print_record(rec, context_metadata_key=context_metadata_key, max_chars=max_chars, as_json=as_json)


def _qdrant_point_id_for_chunk_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def inspect_qdrant(
    *,
    qdrant_storage_path: str,
    collection_name: str,
    chunk_id: str | None,
    limit: int,
    context_metadata_key: str,
    max_chars: int | None,
    as_json: bool,
) -> None:
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(path=qdrant_storage_path)
    if not qdrant.collection_exists(collection_name):
        raise SystemExit(f"Collection does not exist: {collection_name}")

    if chunk_id:
        point_id = _qdrant_point_id_for_chunk_id(chunk_id)
        points = qdrant.retrieve(collection_name=collection_name, ids=[point_id], with_payload=True, with_vectors=False)
    else:
        points, _offset = qdrant.scroll(
            collection_name=collection_name, limit=limit, with_payload=True, with_vectors=False
        )

    for point in points:
        payload = dict(point.payload or {})
        rec = {"chunk_id": payload.get("chunk_id") or str(point.id), "payload": payload}
        _print_record(rec, context_metadata_key=context_metadata_key, max_chars=max_chars, as_json=as_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect stored chunk payload/context for sanity-checking.")
    parser.add_argument("--backend", choices=["milvus", "qdrant"], required=True)
    parser.add_argument("--collection-name", default="rag_chunks")
    parser.add_argument("--chunk-id", default=None, help="Primary chunk id (DocChunk.id). If omitted, prints a sample.")
    parser.add_argument("--limit", type=int, default=5, help="Number of items to sample when --chunk-id is omitted.")
    parser.add_argument("--context-metadata-key", default="context", help="Metadata key where context text is stored.")
    parser.add_argument(
        "--max-chars", type=int, default=400, help="Truncate context to this many chars (use 0 to disable)."
    )
    parser.add_argument("--json", action="store_true", help="Print JSONL records.")

    parser.add_argument("--milvus-uri", default=None, help="Milvus URI or local path (Milvus Lite).")
    parser.add_argument("--qdrant-storage-path", default=None, help="Qdrant local storage path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_chars = None if int(args.max_chars or 0) <= 0 else int(args.max_chars)

    if args.backend == "milvus":
        if not args.milvus_uri:
            raise SystemExit("--milvus-uri is required for --backend=milvus")
        inspect_milvus(
            milvus_uri=args.milvus_uri,
            collection_name=args.collection_name,
            chunk_id=args.chunk_id,
            limit=int(args.limit),
            context_metadata_key=args.context_metadata_key,
            max_chars=max_chars,
            as_json=bool(args.json),
        )
        return

    if args.backend == "qdrant":
        if not args.qdrant_storage_path:
            raise SystemExit("--qdrant-storage-path is required for --backend=qdrant")
        inspect_qdrant(
            qdrant_storage_path=args.qdrant_storage_path,
            collection_name=args.collection_name,
            chunk_id=args.chunk_id,
            limit=int(args.limit),
            context_metadata_key=args.context_metadata_key,
            max_chars=max_chars,
            as_json=bool(args.json),
        )
        return

    raise SystemExit(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
