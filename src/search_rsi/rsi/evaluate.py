from __future__ import annotations

import time
from dataclasses import dataclass, field

from search_rsi.eval.metrics import METRICS
from search_rsi.memory.rewrites import RewriteRules
from search_rsi.pipeline import PipelineConfig, SearchPipeline
from search_rsi.problems.base import Query, SearchProblem

EVAL_DEPTH = 100


@dataclass(frozen=True)
class PipelineScore:
    """Field names match the agentic meta-loop's VariantScore, so the same Archive
    (and its Pareto front) serves both loops."""

    mean_score: float  # the problem's primary metric
    metrics: dict = field(hash=False)  # every metric in METRICS, averaged
    mean_tool_calls: float  # retrieval calls per query
    mean_latency_ms: float
    n_tasks: int
    per_query: dict = field(hash=False, compare=False, repr=False)  # qid -> primary metric


def evaluate_pipeline(
    problem: SearchProblem,
    config: PipelineConfig,
    queries: str | list[Query],
    rewrites: RewriteRules | None = None,
) -> PipelineScore:
    """Run `config` over `queries` (a split name or explicit list) and grade every
    ranking against the problem's qrels. Read-only: evaluation never writes rewrite
    memory, whatever the split (plan.md C4)."""
    if isinstance(queries, str):
        queries = problem.split(queries)
    if not queries:
        raise ValueError(f"{problem.name}: no queries to evaluate")
    pipeline = SearchPipeline(problem.documents, config, rewrites, cache=problem.index_cache)
    totals = {name: 0.0 for name in METRICS}
    per_query: dict[str, float] = {}
    calls = 0
    elapsed = 0.0
    for q in queries:
        start = time.perf_counter()
        response = pipeline.search(q.text, top_k=EVAL_DEPTH)
        elapsed += time.perf_counter() - start
        calls += response.retrieval_calls
        for name, fn in METRICS.items():
            value = fn(response.doc_ids, q.qrels)
            totals[name] += value
            if name == problem.primary_metric:
                per_query[q.qid] = value
    n = len(queries)
    metrics = {name: total / n for name, total in totals.items()}
    return PipelineScore(
        mean_score=metrics[problem.primary_metric],
        metrics=metrics,
        mean_tool_calls=calls / n,
        mean_latency_ms=elapsed * 1000 / n,
        n_tasks=n,
        per_query=per_query,
    )
