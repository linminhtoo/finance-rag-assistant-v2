from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from finrag.eval.schema import EvalItem


def load_jsonl(path: str | Path) -> list[EvalItem]:
    p = Path(path)
    items: list[EvalItem] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(EvalItem.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL at {p}:{line_no}: {e}") from e
    return items


def dump_jsonl(items: Iterable[EvalItem], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json())
            f.write("\n")

