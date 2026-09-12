from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryEntry:
    task_type: str
    domain: str
    strategy: str
    evidence: str  # what observation led to this strategy being recorded


class MemoryStore:
    """Persisted, inspectable cross-run memory (plan.md C3).

    Entries are keyed by (task_type, domain) signature and are plain JSON on disk —
    no gradient updates, no weight checkpoints. Reading never mutates the file;
    writing is append-only per signature (deduplicated on strategy text).
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text())

    def _save(self, entries: list[dict]) -> None:
        self.path.write_text(json.dumps(entries, indent=2))

    def read(self, task_type: str, domain: str) -> list[MemoryEntry]:
        return [
            MemoryEntry(**e)
            for e in self._load()
            if e["task_type"] == task_type and e["domain"] == domain
        ]

    def write(self, entry: MemoryEntry) -> None:
        entries = self._load()
        for e in entries:
            if e["task_type"] == entry.task_type and e["domain"] == entry.domain and e["strategy"] == entry.strategy:
                return  # already recorded
        entries.append(asdict(entry))
        self._save(entries)
