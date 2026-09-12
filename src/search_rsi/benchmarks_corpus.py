from __future__ import annotations

import json
from pathlib import Path

from search_rsi.types import Document, Task

CORPUS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus"


def load_default_corpus() -> list[Document]:
    data = json.loads((CORPUS_DIR / "docs.json").read_text())
    return [Document(doc_id=d["doc_id"], text=d["text"]) for d in data]


def load_gold_tasks() -> list[Task]:
    data = json.loads((CORPUS_DIR / "gold_tasks.json").read_text())
    return [Task(**t) for t in data]


def load_task_splits() -> dict[str, list[Task]]:
    """The frozen three-way split (docs/META_RSI.md Guardrail 2).

    Indices are written once by generate_corpus.py into splits.json, balanced by
    topic chain so every split sees every domain and both task types — not derived
    at random each call, so the split can't silently drift between the inner loop,
    the meta loop, and the final reported eval.
    """
    tasks = load_gold_tasks()
    index_splits = json.loads((CORPUS_DIR / "splits.json").read_text())
    return {name: [tasks[i] for i in idxs] for name, idxs in index_splits.items()}
