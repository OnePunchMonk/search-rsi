from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


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
