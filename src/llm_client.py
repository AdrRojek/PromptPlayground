"""Shared chat client."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

Provider = Literal["openai", "ollama"]

@dataclass
class ChatResult:
    content: str
    latency_ms: int
    provider: Provider
    model: str
    temperature: float
    raw_meta: dict[str, Any]

class LLMError(Exception):
    """Raised when a provider call fails."""
