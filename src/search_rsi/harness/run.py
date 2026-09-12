from __future__ import annotations

from pathlib import Path

from search_rsi.agents import OverlapVerifier, RuleBasedPlanner
from search_rsi.eval import exact_match_score
from search_rsi.memory import MemoryEntry, MemoryStore
from search_rsi.retrieval import Retriever, build_retriever
from search_rsi.retrieval.bm25 import tokenize
from search_rsi.types import Document, Task, TaskResult, ToolCall

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parents[3] / "memory" / "store" / "entries.json"


class Harness:
    """The plan -> retrieve -> verify -> answer -> score -> memory-write run loop.

    Owns the tool-call budget (plan.md C5) and the memory on/off ablation switch
    (plan.md C6). Held-out eval tasks must call run(..., is_training_task=False) so
    they never write memory (plan.md C4).
    """

    def __init__(
        self,
        index: Retriever,
        planner: RuleBasedPlanner,
        verifier: OverlapVerifier,
        memory: MemoryStore,
    ):
        self.index = index
        self.planner = planner
        self.verifier = verifier
        self.memory = memory

    @classmethod
    def local(
        cls,
        documents: list[Document] | None = None,
        memory_path: Path | None = None,
        retriever: str = "bm25",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        verifier_min_overlap: int = 1,
        use_hyde: bool = False,
    ) -> "Harness":
        """Build a harness from documents plus the parameters *and the retrieval
        algorithm* the meta-loop searches over (docs/META_RSI.md) — algorithm choice
        is part of the mutation surface, not a fixed implementation detail. Defaults
        are the hand-written baseline config."""
        from search_rsi.benchmarks_corpus import load_default_corpus

        docs = documents if documents is not None else load_default_corpus()
        return cls(
            index=build_retriever(retriever, docs, bm25_k1=bm25_k1, bm25_b=bm25_b, use_hyde=use_hyde),
            planner=RuleBasedPlanner(),
            verifier=OverlapVerifier(min_overlap=verifier_min_overlap),
            memory=MemoryStore(memory_path or DEFAULT_MEMORY_PATH),
        )

    def run(self, task: Task, memory_enabled: bool = True, is_training_task: bool = True) -> TaskResult:
        memory_entries = self.memory.read(task.task_type, task.domain) if memory_enabled else []
        queries = self.planner.plan_queries(task, memory_entries)

        tool_calls: list[ToolCall] = []
        question_terms = set(tokenize(task.question))
        supported_doc_ids: list[str] = []

        for query in queries:
            if len(tool_calls) >= task.tool_budget:
                break
            hits = self.index.search(query, top_k=3)
            result_doc_ids = [doc_id for doc_id, _ in hits]
            tool_calls.append(ToolCall(kind="bm25_search", query=query, result_doc_ids=result_doc_ids))
            for doc_id in result_doc_ids:
                doc_text = next(d.text for d in self.index.documents if d.doc_id == doc_id)
                if self.verifier.supports(question_terms, doc_text) and doc_id not in supported_doc_ids:
                    supported_doc_ids.append(doc_id)

        answer = supported_doc_ids[0] if supported_doc_ids else ""
        score = exact_match_score(task, answer, supported_doc_ids)

        if memory_enabled and is_training_task and score < 1.0 and len(queries) == 1:
            self.memory.write(
                MemoryEntry(
                    task_type=task.task_type,
                    domain=task.domain,
                    strategy="decompose multi-part questions into sub-queries before retrieval",
                    evidence=f"single-query plan scored {score:.2f} on: {task.question!r}",
                )
            )

        return TaskResult(answer=answer, citations=supported_doc_ids, tool_calls=tool_calls, score=score)
