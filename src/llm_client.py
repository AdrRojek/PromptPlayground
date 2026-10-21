"""Shared chat client for OpenAI and Ollama."""
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

def chat(provider: Provider, model: str, system: str, user: str,
         temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> ChatResult:
    raise LLMError(f"Provider not implemented: {provider}")
