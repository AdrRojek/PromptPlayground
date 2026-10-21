"""Shared chat client for OpenAI and Ollama."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
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


def chat(
    provider: Provider,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    *,
    openai_api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> ChatResult:
    if provider == "openai":
        return _chat_openai(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key,
        )
    if provider == "ollama":
        return _chat_ollama(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=ollama_base_url,
        )
    raise LLMError(f"Unknown provider: {provider}")


def _chat_openai(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
) -> ChatResult:
    if not api_key:
        raise LLMError("OpenAI API key is missing. Set OPENAI_API_KEY or enter a key in the sidebar.")

    client = OpenAI(api_key=api_key)
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - surface provider errors to UI
        raise LLMError(f"OpenAI request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    usage = getattr(response, "usage", None)
    raw_meta: dict[str, Any] = {
        "id": getattr(response, "id", None),
        "finish_reason": choice.finish_reason,
        "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
    }
    return ChatResult(
        content=content,
        latency_ms=latency_ms,
        provider="openai",
        model=model,
        temperature=temperature,
        raw_meta=raw_meta,
    )


def _chat_ollama(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
) -> ChatResult:
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001 - surface provider errors to UI
        raise LLMError(f"Ollama request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()
    raw_meta = {
        "total_duration": data.get("total_duration"),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "done_reason": data.get("done_reason"),
    }
    return ChatResult(
        content=content,
        latency_ms=latency_ms,
        provider="ollama",
        model=model,
        temperature=temperature,
        raw_meta=raw_meta,
    )
