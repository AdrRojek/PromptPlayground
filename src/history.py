"""JSONL history persistence for prompt runs."""
from __future__ import annotations
from dataclasses import asdict, dataclass
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
