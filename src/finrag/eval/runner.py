import concurrent.futures
import json
import os
import multiprocessing
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from finrag.dataclasses import TopChunk
from finrag.generation_controls import AnswerStyle, GenerationSettings, resolve_generation_settings
from finrag.eval.schema import EvalGeneration, EvalQuery, RetrievedChunk, EvalKind
from finrag.main import RAGService


@dataclass(frozen=True)
class RunConfig:
    mode: str = "normal"

    # Optional overrides (fall back to preset when None).
    top_k_retrieve: int | None = None
    top_k_rerank: int | None = None
    draft_max_tokens: int | None = None
    final_max_tokens: int | None = None
    enable_rerank: bool | None = None
    enable_refine: bool | None = None
    answer_style: AnswerStyle | None = None
    draft_temperature: float | None = None

    # Parallelism. (Latency doesn't matter for offline eval runs.)
    concurrency: int = 8

    # Output controls.
    max_chunks: int = 50
    chunk_text_chars: int = 2000
    chunk_context_chars: int = 2000

    def resolved_settings(self) -> GenerationSettings:
        return resolve_generation_settings(
            mode=self.mode,
            top_k_retrieve=self.top_k_retrieve,
            top_k_rerank=self.top_k_rerank,
            draft_max_tokens=self.draft_max_tokens,
            final_max_tokens=self.final_max_tokens,
            enable_rerank=self.enable_rerank,
            enable_refine=self.enable_refine,
            answer_style=self.answer_style,
            draft_temperature=self.draft_temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_WORKER_SERVICE: Any | None = None
_WORKER_SETTINGS: GenerationSettings | None = None
_WORKER_CFG: RunConfig | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    s = str(text)
    if limit <= 0 or len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def _to_retrieved_chunk(chunk: TopChunk, cfg: RunConfig) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page_no=chunk.page_no,
        headings=list(chunk.headings or []),
        score=float(chunk.score),
        source=chunk.source,
        preview=_truncate(chunk.preview, 400),
        text=_truncate(chunk.text, cfg.chunk_text_chars),
        context=_truncate(chunk.context, cfg.chunk_context_chars),
        metadata=chunk.metadata,
    )


def _run_one(
    service: RAGService,
    query_id: str,
    kind: EvalKind,
    question: str,
    settings: GenerationSettings,
    cfg: RunConfig,
) -> tuple[EvalGeneration, float, bool]:
    t0 = time.perf_counter()
    created = _utcnow()
    try:
        resp = service.answer_question(question, settings)
        chunks = [_to_retrieved_chunk(tc, cfg) for tc in (resp.top_chunks or [])[: cfg.max_chunks]]
        gen = EvalGeneration(
            query_id=query_id,
            kind=kind,
            question=question,
            created_at=created,
            settings={
                "mode": settings.mode,
                "top_k_retrieve": settings.top_k_retrieve,
                "top_k_rerank": settings.top_k_rerank,
                "draft_max_tokens": settings.draft_max_tokens,
                "final_max_tokens": settings.final_max_tokens,
                "enable_rerank": settings.enable_rerank,
                "enable_refine": settings.enable_refine,
                "answer_style": settings.answer_style,
                "draft_temperature": settings.draft_temperature,
                "concurrency": max(1, int(cfg.concurrency)),
            },
            draft_answer=resp.draft_answer,
            final_answer=resp.final_answer,
            top_chunks=chunks,
        )
        ok = True
    except Exception as exc:  # noqa: BLE001
        gen = EvalGeneration(
            query_id=query_id,
            kind=kind,
            question=question,
            created_at=created,
            settings={"mode": settings.mode, "concurrency": max(1, int(cfg.concurrency))},
            error=str(exc),
        )
        ok = False
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    gen.timing_ms["total_ms"] = ms
    return gen, ms, ok


def _worker_init(cfg_dict: dict[str, Any], storage_path: str | None) -> None:
    global _WORKER_SERVICE, _WORKER_SETTINGS, _WORKER_CFG

    if storage_path is not None:
        os.environ["QDRANT_STORAGE_PATH"] = storage_path

    cfg = RunConfig(**cfg_dict)
    settings = cfg.resolved_settings()

    import finrag.main as main

    _WORKER_SERVICE = main.rag_service
    _WORKER_SETTINGS = settings
    _WORKER_CFG = cfg


def _worker_run_one(query_id: str, kind: EvalKind, question: str) -> tuple[str, float, bool]:
    if _WORKER_SERVICE is None or _WORKER_SETTINGS is None or _WORKER_CFG is None:  # pragma: no cover
        raise RuntimeError("Worker not initialized")
    gen, ms, ok = _run_one(_WORKER_SERVICE, query_id, kind, question, _WORKER_SETTINGS, _WORKER_CFG)
    return gen.model_dump_json(), ms, ok


def run_generation(
    queries: Iterable[EvalQuery],
    *,
    out_jsonl: str | Path,
    cfg: RunConfig,
    storage_path: str | None = None,
) -> dict[str, Any]:
    """
    Run `EvalQuery`s through the app's `RAGService.answer_question()` pipeline and
    write `EvalGeneration` JSONL.
    """
    p = Path(out_jsonl)
    p.parent.mkdir(parents=True, exist_ok=True)

    # `finrag.main` constructs a global `rag_service` at import time.
    # Reuse it here to avoid initializing the whole stack twice.
    if storage_path is not None:
        os.environ["QDRANT_STORAGE_PATH"] = storage_path
    queries_list = list(queries)
    query_specs: list[tuple[str, EvalKind, str]] = [(q.id, q.kind, q.question) for q in queries_list]
    n = 0
    n_ok = 0
    n_err = 0
    total_ms = 0.0
    concurrency = max(1, int(cfg.concurrency))
    wall_t0 = time.perf_counter()

    if concurrency <= 1 or len(query_specs) <= 1:
        import finrag.main as main

        service = main.rag_service
        settings = cfg.resolved_settings()
        with p.open("w", encoding="utf-8") as f:
            for query_id, kind, question in query_specs:
                gen, ms, ok = _run_one(service, query_id, kind, question, settings, cfg)
                n += 1
                total_ms += ms
                if ok:
                    n_ok += 1
                else:
                    n_err += 1
                f.write(gen.model_dump_json())
                f.write("\n")
    else:
        with p.open("w", encoding="utf-8") as f:
            pending: dict[int, str] = {}
            next_to_write = 0

            mp_ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(concurrency, len(query_specs)),
                mp_context=mp_ctx,
                initializer=_worker_init,
                initargs=(cfg.to_dict(), storage_path),
            ) as ex:
                fut_to_idx = {
                    ex.submit(_worker_run_one, query_id, kind, question): i
                    for i, (query_id, kind, question) in enumerate(query_specs)
                }

                for fut in concurrent.futures.as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    query_id, kind, question = query_specs[i]
                    try:
                        line, ms, ok = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        created = _utcnow()
                        gen = EvalGeneration(
                            query_id=query_id,
                            kind=kind,
                            question=question,
                            created_at=created,
                            settings={"mode": cfg.mode, "concurrency": concurrency},
                            error=f"Worker failed: {exc}",
                        )
                        gen.timing_ms["total_ms"] = 0.0
                        line, ms, ok = gen.model_dump_json(), 0.0, False

                    n += 1
                    total_ms += ms
                    if ok:
                        n_ok += 1
                    else:
                        n_err += 1

                    pending[i] = line
                    while next_to_write in pending:
                        f.write(pending.pop(next_to_write))
                        f.write("\n")
                        next_to_write += 1

            if pending:  # pragma: no cover
                for i in sorted(pending):
                    f.write(pending[i])
                    f.write("\n")

    wall_t1 = time.perf_counter()
    wall_total_ms = (wall_t1 - wall_t0) * 1000.0

    summary = {
        "n": n,
        "n_ok": n_ok,
        "n_err": n_err,
        "avg_total_ms": (total_ms / n) if n else 0.0,
        "wall_total_ms": wall_total_ms,
        "settings": cfg.to_dict(),
    }
    return summary


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
