from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from finrag.llm_clients import LLMClient


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def ndjson_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass
class TextDeltaBatcher:
    flush_tokens: int
    flush_chars: int
    flush_interval_ms: int

    _buf: str = ""
    _last_flush_ms: int = 0

    @classmethod
    def from_env(cls) -> "TextDeltaBatcher":
        return cls(
            flush_tokens=_env_int("FINRAG_STREAM_FLUSH_TOKENS", 24),
            flush_chars=_env_int("FINRAG_STREAM_FLUSH_CHARS", 120),
            flush_interval_ms=_env_int("FINRAG_STREAM_FLUSH_INTERVAL_MS", 120),
        )

    def add(self, delta: str) -> None:
        self._buf += delta or ""

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def pop_ready(self) -> str | None:
        if not self._buf:
            return None

        now = self._now_ms()
        should_flush = False
        if self.flush_chars > 0 and len(self._buf) >= self.flush_chars:
            should_flush = True
        if self.flush_tokens > 0 and _approx_tokens(self._buf) >= self.flush_tokens:
            should_flush = True
        if self.flush_interval_ms > 0 and self._last_flush_ms and (now - self._last_flush_ms) >= self.flush_interval_ms:
            should_flush = True

        if not should_flush:
            return None

        out = self._buf
        self._buf = ""
        self._last_flush_ms = now
        return out

    def pop_all(self) -> str | None:
        if not self._buf:
            return None
        out = self._buf
        self._buf = ""
        self._last_flush_ms = self._now_ms()
        return out


async def iter_chat_deltas(
    llm: LLMClient,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    is_cancelled: Callable[[], bool],
    set_cancelled: Callable[[], None],
    is_disconnected: Callable[[], "asyncio.Future[bool]"] | Callable[[], Any],
) -> AsyncIterator[str]:
    """
    Bridges a blocking provider stream to an async iterator.

    Expected contract for `llm`:
      - `chat_stream(messages, temperature=...) -> Iterator[str]` (preferred), OR
      - `chat(messages, temperature=...) -> str` (fallback).
    """

    async def disconnected() -> bool:
        try:
            out = is_disconnected()
            if asyncio.iscoroutine(out):
                return bool(await out)
            return bool(out)
        except Exception:
            return False

    if not hasattr(llm, "chat_stream"):
        # Fallback: no token streaming available, so emit one big delta.
        text = await asyncio.to_thread(llm.chat, messages, temperature)  # type: ignore[attr-defined]
        if text:
            yield str(text)
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()

    def worker() -> None:
        try:
            it: Iterator[str] = llm.chat_stream(messages, temperature=temperature)  # type: ignore[attr-defined]
            for delta in it:
                if is_cancelled():
                    break
                if delta:
                    loop.call_soon_threadsafe(queue.put_nowait, str(delta))
        except BaseException as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            if await disconnected():
                set_cancelled()

            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

            if is_cancelled():
                break
    finally:
        set_cancelled()
        try:
            await asyncio.wait_for(thread_task, timeout=1.0)
        except Exception:
            pass


def stream_draft_enabled() -> bool:
    return _env_bool("FINRAG_STREAM_DRAFT", default=True)


def stream_chunks_preview_chars() -> int:
    return _env_int("FINRAG_STREAM_CHUNKS_PREVIEW_CHARS", 260)


def stream_chunks_max() -> int:
    return _env_int("FINRAG_STREAM_CHUNKS_MAX", 30)

