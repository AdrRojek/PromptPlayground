"""JSONL history persistence for prompt runs."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def append_entry(path: Path, entry: HistoryEntry) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

def make_entry(*, provider, model, temperature, max_tokens, system_prompt,
               user_prompt, response, latency_ms, mode="single", label=None, meta=None):
    return HistoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider, model=model, temperature=temperature,
        max_tokens=max_tokens, system_prompt=system_prompt, user_prompt=user_prompt,
        response=response, latency_ms=latency_ms, mode=mode, label=label, meta=meta,
    )
