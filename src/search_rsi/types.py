from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    # Structured fields (price, brand, date, ...) that filters and boosts read.
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    def indexed_text(self, title_boost: int = 0) -> str:
        """Title repeated `1 + title_boost` times ahead of the body — a BM25F-lite
        field weighting that works with any single-field scorer."""
        if not self.title:
            return self.text
        return " ".join([self.title] * (1 + title_boost) + [self.text])


@dataclass(frozen=True)
class Task:
    question: str
    gold_doc_ids: list[str]
    gold_answer: str
    tool_budget: int
    task_type: str = "lookup"
    domain: str = "general"


@dataclass
class ToolCall:
    kind: str
    query: str
    result_doc_ids: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    answer: str
    citations: list[str]
    tool_calls: list[ToolCall]
    score: float

    @property
    def tool_calls_used(self) -> int:
        return len(self.tool_calls)
