"""Helpers for side-by-side prompt comparison runs."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import ChatResult, LLMError, Provider, chat


@dataclass
class CompareSettings:
    label: str
    provider: Provider
    model: str
    temperature: float
    max_tokens: int


@dataclass
class CompareOutcome:
    settings: CompareSettings
    result: ChatResult | None
    error: str | None


def run_compare(
    *,
    system: str,
    user: str,
    left: CompareSettings,
    right: CompareSettings,
    openai_api_key: str | None,
    ollama_base_url: str,
) -> tuple[CompareOutcome, CompareOutcome]:
    return (
        _run_one(system, user, left, openai_api_key, ollama_base_url),
        _run_one(system, user, right, openai_api_key, ollama_base_url),
    )


def _run_one(
    system: str,
    user: str,
    settings: CompareSettings,
    openai_api_key: str | None,
    ollama_base_url: str,
) -> CompareOutcome:
    try:
        result = chat(
            provider=settings.provider,
            model=settings.model,
            system=system,
            user=user,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=openai_api_key,
            ollama_base_url=ollama_base_url,
        )
        return CompareOutcome(settings=settings, result=result, error=None)
    except LLMError as exc:
        return CompareOutcome(settings=settings, result=None, error=str(exc))
