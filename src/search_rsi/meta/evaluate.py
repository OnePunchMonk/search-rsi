from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from search_rsi.harness import Harness
from search_rsi.meta.config import HarnessConfig
from search_rsi.types import Task


@dataclass(frozen=True)
class VariantScore:
    mean_score: float
    mean_tool_calls: float
    mean_latency_ms: float
    n_tasks: int


def evaluate_variant(config: HarnessConfig, tasks: list[Task]) -> VariantScore:
    """The meta-loop's sandbox: build a harness from `config` and run it against a
    fixed task set with memory disabled, so the numbers measure what the config
    change did, not noise from the inner-loop strategy memory (docs/META_RSI.md).

    Uses a throwaway memory path per call — a meta-eval run must never read or write
    the persistent strategy store (that store belongs to the inner loop only).

    Latency is measured end-to-end per task (index build is amortized outside the
    timed loop, matching a warm production server, not a cold start) — it's the
    other half of the actual product goal alongside score: "given a search problem,
    find the best-scoring, lowest-latency config," not best-scoring alone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness.local(
            memory_path=Path(tmp) / "entries.json",
            retriever=config.retriever,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
            verifier_min_overlap=config.verifier_min_overlap,
            use_hyde=config.use_hyde,
        )
        scores = []
        calls = []
        latencies_ms = []
        for task in tasks:
            start = time.perf_counter()
            result = harness.run(task, memory_enabled=False, is_training_task=False)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            scores.append(result.score)
            calls.append(result.tool_calls_used)
        return VariantScore(
            mean_score=sum(scores) / len(scores),
            mean_tool_calls=sum(calls) / len(calls),
            mean_latency_ms=sum(latencies_ms) / len(latencies_ms),
            n_tasks=len(tasks),
        )
