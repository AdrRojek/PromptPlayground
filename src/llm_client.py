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

def chat(provider, model, system, user, temperature=0.7, max_tokens=1024,
         *, openai_api_key=None, ollama_base_url="http://localhost:11434"):
    if provider == "openai":
        return _chat_openai(model, system, user, temperature, max_tokens, openai_api_key)
    if provider == "ollama":
        return _chat_ollama(model, system, user, temperature, max_tokens, ollama_base_url)
    raise LLMError(f"Unknown provider: {provider}")

def _chat_openai(model, system, user, temperature, max_tokens, api_key):
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
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
    except Exception as exc:
        raise LLMError(f"OpenAI request failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    return ChatResult(content, latency_ms, "openai", model, temperature, {
        "id": getattr(response, "id", None),
        "finish_reason": choice.finish_reason,
    })

def _chat_ollama(model, system, user, temperature, max_tokens, base_url):
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)
    content = ((data.get("message") or {}).get("content") or "").strip()
    return ChatResult(content, latency_ms, "ollama", model, temperature, {
        "total_duration": data.get("total_duration"),
        "eval_count": data.get("eval_count"),
    })
