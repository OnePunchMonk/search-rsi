from __future__ import annotations

from search_rsi.retrieval.bm25 import tokenize
from search_rsi.types import Task


def exact_match_score(task: Task, answer: str, citations: list[str]) -> float:
    """The one executed score (plan.md C1).

    Combines answer correctness against the real gold answer with citation support
    against the real gold doc ids, both computed from the task's actual data rather
    than any formula that ignores it.
    """
    answer_correct = tokenize(answer) == tokenize(task.gold_answer)
    citation_recall = (
        len(set(citations) & set(task.gold_doc_ids)) / len(task.gold_doc_ids)
        if task.gold_doc_ids
        else 1.0
    )
    return 0.7 * float(answer_correct) + 0.3 * citation_recall
