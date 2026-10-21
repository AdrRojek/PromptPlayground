"""Shared chat client for OpenAI and Ollama."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Literal
from openai import OpenAI

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

def chat(provider, model, system, user, temperature=0.7, max_tokens=1024,
         *, openai_api_key=None, ollama_base_url="http://localhost:11434"):
    if provider != "openai":
        raise LLMError(f"Provider not implemented: {provider}")
    if not openai_api_key:
        raise LLMError("OpenAI API key is missing.")
    client = OpenAI(api_key=openai_api_key)
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    content = (response.choices[0].message.content or "").strip()
    return ChatResult(content, latency_ms, "openai", model, temperature, {"id": response.id})
