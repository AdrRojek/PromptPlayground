"""Helpers for side-by-side prompt comparison runs."""
from __future__ import annotations
from dataclasses import dataclass
from src.llm_client import ChatResult, Provider

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
