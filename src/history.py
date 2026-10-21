"""JSONL history persistence for prompt runs."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass
class HistoryEntry:
    timestamp: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    user_prompt: str
    response: str
    latency_ms: int
    mode: str = "single"
    label: str | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def append_entry(path: Path, entry: HistoryEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
