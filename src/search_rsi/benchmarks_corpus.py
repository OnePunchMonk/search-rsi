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
