"""Standard ranking metrics — the fixed, untouchable grader for search problems.

Every number that drives an accept/reject or promote decision in the search
pipeline's RSI loops is one of these, computed against a problem's real graded
relevance judgments (qrels). Nothing in `PipelineConfig` or the rewrite memory can
reach this module (docs/META_RSI.md, Guardrail 1).
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

Qrels = Mapping[str, int]


def dcg(gains: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], qrels: Qrels, k: int = 10) -> float:
    ideal = dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([qrels.get(d, 0) for d in ranked[:k]]) / ideal


def mrr_at_k(ranked: Sequence[str], qrels: Qrels, k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], qrels: Qrels, k: int = 10) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def success_at_k(ranked: Sequence[str], qrels: Qrels, k: int = 1) -> float:
    """1.0 if a maximally relevant document is in the top k (navigational search:
    'did the thing the user meant come first?')."""
    top_grade = max(qrels.values(), default=0)
    if top_grade <= 0:
        return 0.0
    return float(any(qrels.get(d, 0) == top_grade for d in ranked[:k]))


METRICS: dict[str, Callable[[Sequence[str], Qrels], float]] = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
    "recall@10": lambda r, q: recall_at_k(r, q, 10),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "success@1": lambda r, q: success_at_k(r, q, 1),
}


def score_ranking(ranked: Sequence[str], qrels: Qrels) -> dict[str, float]:
    return {name: fn(ranked, qrels) for name, fn in METRICS.items()}
