"""Environment and default settings for Prompt Playground."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
HISTORY_PATH = DATA_DIR / "history.jsonl"

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "o4-mini",
]


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    default_temperature: float
    default_max_tokens: int
    history_path: Path

    @property
    def preferred_provider(self) -> str:
        return "openai" if self.openai_api_key else "ollama"


def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=DEFAULT_OPENAI_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_URL.rstrip("/"),
        ollama_model=DEFAULT_OLLAMA_MODEL,
        default_temperature=DEFAULT_TEMPERATURE,
        default_max_tokens=DEFAULT_MAX_TOKENS,
        history_path=HISTORY_PATH,
    )
