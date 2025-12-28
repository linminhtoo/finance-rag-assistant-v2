import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, cast

from finrag.dataclasses import DocChunk
from finrag.llm_clients import LLMClient

ContextBuilder = Callable[[DocChunk], str]


def situate_context(
    llm: LLMClient,
    *,
    context: str,
    chunk: str,
    temperature: float = 0.0,
    max_context_chars: int = 20_000,
    max_chunk_chars: int = 6_000,
) -> str:
    """
    Ask an LLM to produce a short "situating" context for `chunk` given some
    surrounding `context`.

    The returned string is meant to be stored in chunk metadata and appended to
    the chunk text before embedding/indexing.
    """

    context = (context or "").strip()
    chunk = (chunk or "").strip()
    if not context or not chunk:
        return ""

    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars].rstrip() + "\n\n[TRUNCATED]"
    if max_chunk_chars > 0 and len(chunk) > max_chunk_chars:
        chunk = chunk[:max_chunk_chars].rstrip() + "\n\n[TRUNCATED]"

    prompt = (
        "You are helping improve vector search retrieval.\n"
        "Given some context and a chunk, write a short, specific context that situates "
        "the chunk within the larger context.\n"
        "Return ONLY the context. Do not include quotes, headings, or preamble.\n\n"
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        "<chunk>\n"
        f"{chunk}\n"
        "</chunk>\n"
    )

    try:
        out = llm.chat([{"role": "user", "content": prompt}], temperature=temperature)
    except Exception as exc:  # noqa: BLE001 - allow in-situate fallback
        raise RuntimeError(f"LLM call failed in situate_context(): {exc!r}") from exc
    return (out or "").strip()


def context_builder_from_metadata(*, key: str = "context") -> ContextBuilder:
    def _builder(chunk: DocChunk) -> str:
        meta = chunk.metadata or {}
        value = meta.get(key)
        return str(value) if value else ""

    return _builder


def apply_context_strategy(
    chunks: list[DocChunk],
    *,
    strategy: str,
    neighbor_window: int = 1,
    doc_text: str | None = None,
    metadata_key: str = "context",
    llm_for_context: LLMClient | None = None,
    temperature: float = 0.0,
    max_concurrency: int = 32,
    skip_if_exists: bool = True,
) -> None:
    """
    Apply a context strategy by generating an LLM-situated context per chunk and
    storing it in chunk metadata.

    Parameters
    ----------
    chunks : list[DocChunk]
        Chunks to annotate with context.
        IMPORTANT: the chunks must belong to the same document and be in document order.
    strategy : str
        Context strategy ("none", "document", "neighbors", "metadata").
        - "document": provide the full document text to the LLM, per chunk.
        - "neighbors": provide neighboring chunk text to the LLM, per chunk.
        - "metadata": assume the context is already present in metadata.
    neighbor_window : int, optional
        Window size when strategy="neighbors".
    doc_text : str | None, optional
        Full document text for strategy="document". Falls back to joining chunk text.
    metadata_key : str, optional
        Metadata key to store the LLM-situated context.
    llm_for_context : LLMClient | None, optional
        LLM client used to generate situated context.
    temperature : float, optional
        LLM temperature for the situating call (default: 0.0).
    max_concurrency : int, optional
        Max parallel LLM calls for context generation (default: 32).
    skip_if_exists : bool, optional
        If True, do not overwrite an existing `metadata_key` value.
    """

    log = logging.getLogger(__name__)
    strategy = strategy.strip().lower()
    if strategy == "none":
        return
    if strategy == "metadata":
        # Assume context is already in metadata
        return
    if llm_for_context is None:
        raise ValueError(f"llm_for_context must be provided when strategy={strategy!r}")

    if strategy == "document":

        def _apply_document() -> None:
            nonlocal doc_text
            if doc_text is None:
                doc_text = "\n\n".join(ch.text for ch in chunks)
            doc_text = cast(str, doc_text)
            max_workers = max(1, int(max_concurrency))
            work_items: list[tuple[int, DocChunk]] = []
            for idx, ch in enumerate(chunks):
                meta = ch.metadata or {}
                if skip_if_exists and meta.get(metadata_key):
                    continue
                work_items.append((idx, ch))

            def _run_item(item: tuple[int, DocChunk]) -> tuple[int, str]:
                if doc_text is None:
                    raise ValueError("doc_text is None in _run_item")
                idx, ch = item
                situated = situate_context(llm_for_context, context=doc_text, chunk=ch.text, temperature=temperature)
                return idx, situated

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_item, item): item for item in work_items}
                for fut in as_completed(futures):
                    _idx, ch = futures[fut]
                    try:
                        _idx2, situated = fut.result()
                    except Exception as exc:  # noqa: BLE001 - surface LLM failures as warnings
                        log.warning("Failed to situate context for chunk_id=%s: %r", ch.id, exc)
                        continue
                    situated = (situated or "").strip()
                    if not situated:
                        continue
                    meta = ch.metadata or {}
                    meta[metadata_key] = situated
                    ch.metadata = meta

        _apply_document()
        return

    if strategy == "neighbors":

        def _apply_neighbors() -> None:
            window = max(0, int(neighbor_window))
            if window == 0:
                return
            texts = [ch.text for ch in chunks]
            max_workers = max(1, int(max_concurrency))
            work_items: list[tuple[int, DocChunk, str]] = []
            for idx, ch in enumerate(chunks):
                meta = ch.metadata or {}
                if skip_if_exists and meta.get(metadata_key):
                    continue
                before = texts[max(0, idx - window) : idx]
                after = texts[idx + 1 : idx + 1 + window]
                parts: list[str] = []
                if before:
                    parts.append("Previous chunks:\n" + "\n\n".join(before))
                if after:
                    parts.append("Next chunks:\n" + "\n\n".join(after))
                neighbor_context = "\n\n".join(p for p in parts if p.strip())
                if not neighbor_context:
                    continue
                work_items.append((idx, ch, neighbor_context))

            def _run_item(item: tuple[int, DocChunk, str]) -> tuple[int, str]:
                idx, ch, neighbor_context = item
                situated = situate_context(
                    llm_for_context, context=neighbor_context, chunk=ch.text, temperature=temperature
                )
                return idx, situated

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_item, item): item for item in work_items}
                for fut in as_completed(futures):
                    _idx, ch, _neighbor_context = futures[fut]
                    try:
                        _idx2, situated = fut.result()
                    except Exception as exc:  # noqa: BLE001 - surface LLM failures as warnings
                        log.warning("Failed to situate neighbor context for chunk_id=%s: %r", ch.id, exc)
                        continue
                    situated = (situated or "").strip()
                    if not situated:
                        continue
                    meta = ch.metadata or {}
                    meta[metadata_key] = situated
                    ch.metadata = meta

        _apply_neighbors()
        return

    raise ValueError(f"Unknown context strategy: {strategy}")
