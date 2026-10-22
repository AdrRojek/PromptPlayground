"""Helpers for side-by-side prompt comparison runs."""
from __future__ import annotations
from dataclasses import dataclass
from src.llm_client import Provider

@dataclass
class CompareSettings:
    label: str
    provider: Provider
    model: str
    temperature: float
    max_tokens: int
