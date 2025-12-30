from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _json_default(obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def load_jsonl(path: str | Path, model: type[T]) -> list[T]:
    p = Path(path)
    items: list[T] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(model.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL at {p}:{line_no}: {e}") from e
    return items


def dump_jsonl(items: Iterable[BaseModel | dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, BaseModel):
                f.write(item.model_dump_json())
            else:
                import json

                f.write(json.dumps(item, ensure_ascii=False, default=_json_default))
            f.write("\n")
