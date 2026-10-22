"""JSONL history persistence for prompt runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_entry(path: Path, entry: HistoryEntry) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def make_entry(
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    response: str,
    latency_ms: int,
    mode: str = "single",
    label: str | None = None,
    meta: dict[str, Any] | None = None,
) -> HistoryEntry:
    return HistoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        latency_ms=latency_ms,
        mode=mode,
        label=label,
        meta=meta,
    )


def load_entries(
    path: Path,
    *,
    provider: str | None = None,
    limit: int | None = 100,
) -> list[HistoryEntry]:
    if not path.exists():
        return []

    entries: list[HistoryEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = HistoryEntry(
                timestamp=raw.get("timestamp", ""),
                provider=raw.get("provider", ""),
                model=raw.get("model", ""),
                temperature=float(raw.get("temperature", 0)),
                max_tokens=int(raw.get("max_tokens", 0)),
                system_prompt=raw.get("system_prompt", ""),
                user_prompt=raw.get("user_prompt", ""),
                response=raw.get("response", ""),
                latency_ms=int(raw.get("latency_ms", 0)),
                mode=raw.get("mode", "single"),
                label=raw.get("label"),
                meta=raw.get("meta"),
            )
            if provider and entry.provider != provider:
                continue
            entries.append(entry)

    entries.reverse()
    if limit is not None:
        return entries[:limit]
    return entries
