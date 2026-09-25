"""The outer RSI loop for search pipelines: evolve the pipeline itself.

Same guardrails as the agentic meta-loop (docs/META_RSI.md): a closed mutation
surface (`PipelineConfig`), accept/reject decisions made on `meta_val` only, a DGM-
style archive of accepted variants with Pareto-aware acceptance, and a random-search
control arm with the identical evaluation budget. What is new is that candidates
which enable learned rewrites get their own rules learned on `train` with their own
pipeline — the inner loop runs inside the outer loop's evaluation, so memory and
pipeline co-adapt instead of the memory being tuned to a pipeline nobody deploys.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable

from search_rsi.memory.rewrites import RewriteRules
from search_rsi.meta.archive import Archive, ArchiveEntry
from search_rsi.pipeline import BASELINE_PIPELINE, PipelineConfig
from search_rsi.problems.base import SearchProblem
from search_rsi.retrieval import pipeline_retriever_names
from search_rsi.rsi.evaluate import PipelineScore, evaluate_pipeline
from search_rsi.rsi.learn import learn_rewrites
from search_rsi.text import ANALYZERS

OPTIMIZE_SPLIT = "meta_val"


def _mutators(retrievers: tuple[str, ...]) -> dict[str, Callable[[PipelineConfig, random.Random], PipelineConfig]]:
    def other(values, current, rng):
        return rng.choice([v for v in values if v != current])

    return {
        "analyzer": lambda c, r: replace(c, analyzer=other(tuple(ANALYZERS), c.analyzer, r)),
        "retriever": lambda c, r: replace(c, retriever=other(retrievers, c.retriever, r)),
        "fusion_retriever": lambda c, r: replace(
            c, fusion_retriever=other(("none",) + tuple(x for x in retrievers if x != c.retriever), c.fusion_retriever, r)),
        "bm25_k1": lambda c, r: replace(c, bm25_k1=round(min(3.0, max(0.1, c.bm25_k1 + r.uniform(-0.6, 0.6))), 2)),
        "bm25_b": lambda c, r: replace(c, bm25_b=round(min(1.0, max(0.0, c.bm25_b + r.uniform(-0.25, 0.25))), 2)),
        "rrf_k": lambda c, r: replace(c, rrf_k=other((10, 30, 60, 100), c.rrf_k, r)),
        "title_boost": lambda c, r: replace(c, title_boost=other((0, 1, 2, 3), c.title_boost, r)),
        "spell_correct": lambda c, r: replace(c, spell_correct=not c.spell_correct),
        "prf_terms": lambda c, r: replace(c, prf_terms=other((0, 3, 5, 10), c.prf_terms, r)),
        "filter_mode": lambda c, r: replace(c, filter_mode=other(("none", "soft", "hard"), c.filter_mode, r)),
        "recency_weight": lambda c, r: replace(c, recency_weight=other((0.0, 0.2, 0.4, 0.6, 0.8), c.recency_weight, r)),
        "rerank_weight": lambda c, r: replace(c, rerank_weight=other((0.0, 0.2, 0.4, 0.6), c.rerank_weight, r)),
        "use_learned_rewrites": lambda c, r: replace(c, use_learned_rewrites=not c.use_learned_rewrites),
    }


def propose_pipeline_mutation(parent: PipelineConfig, rng: random.Random) -> PipelineConfig:
    """Change one stage (70%) or two (30%) — small, attributable steps."""
    mutators = _mutators(pipeline_retriever_names())
    child = parent
    for name in rng.sample(sorted(mutators), 2 if rng.random() < 0.3 else 1):
        child = mutators[name](child, rng)
    return child


def random_pipeline_config(rng: random.Random) -> PipelineConfig:
    """The control arm: every stage drawn independently of any evaluation."""
    retrievers = pipeline_retriever_names()
    retriever = rng.choice(retrievers)
    return PipelineConfig(
        analyzer=rng.choice(tuple(ANALYZERS)),
        retriever=retriever,
        fusion_retriever=rng.choice(("none",) + tuple(x for x in retrievers if x != retriever)),
        bm25_k1=round(rng.uniform(0.3, 2.5), 2),
        bm25_b=round(rng.uniform(0.0, 1.0), 2),
        rrf_k=rng.choice((10, 30, 60, 100)),
        title_boost=rng.choice((0, 1, 2, 3)),
        spell_correct=rng.random() < 0.5,
        prf_terms=rng.choice((0, 3, 5, 10)),
        filter_mode=rng.choice(("none", "soft", "hard")),
        recency_weight=rng.choice((0.0, 0.2, 0.4, 0.6, 0.8)),
        rerank_weight=rng.choice((0.0, 0.2, 0.4, 0.6)),
        use_learned_rewrites=rng.random() < 0.5,
    )


def _tournament(archive: Archive, rng: random.Random, size: int = 2) -> ArchiveEntry:
    """Sample `size` archive members uniformly, keep the best: selection pressure
    toward good parents without collapsing onto the champion (DGM-style)."""
    contenders = [rng.choice(archive.entries) for _ in range(size)]
    return max(contenders, key=lambda e: e.score.mean_score)


def _admit(archive: Archive, champion: ArchiveEntry, candidate: ArchiveEntry,
           parent: ArchiveEntry | None, floor: float, stepping_stones: bool) -> tuple[bool, bool]:
    """Returns (admitted, new_champion).

    The champion changes only through the strict Pareto gate against the current
    champion (Archive.passes_gate, significance floor included). With
    `stepping_stones`, a candidate that clearly beats *its own parent* still enters
    the archive — the DGM finding that a variant worse than the champion can be the
    better parent for the next mutation — but can never become champion that way.
    """
    if Archive.passes_gate(candidate, champion, floor):
        archive.entries.append(candidate)
        return True, True
    if (stepping_stones and parent is not None
            and candidate.score.mean_score > parent.score.mean_score + floor / 2
            and all(e.config != candidate.config for e in archive.entries)):
        archive.entries.append(candidate)
        return True, False
    return False, False


class RuleBook:
    """Rules learned on `train`, one rule set per pipeline config (memoized), seeded
    from any persisted rules so learning continues across runs."""

    def __init__(self, problem: SearchProblem, seed_rules: RewriteRules | None = None):
        self.problem = problem
        self.seed_rules = seed_rules or RewriteRules()
        self._by_config: dict[PipelineConfig, RewriteRules] = {}

    def rules_for(self, config: PipelineConfig) -> RewriteRules | None:
        if not config.use_learned_rewrites:
            return None
        if config not in self._by_config:
            self._by_config[config] = learn_rewrites(self.problem, config, self.seed_rules).rules
        return self._by_config[config]


@dataclass
class OptimizeResult:
    problem: str
    strategy: str
    champion: ArchiveEntry
    baseline: PipelineScore
    archive: Archive = field(repr=False)
    rulebook: RuleBook = field(repr=False)
    n_accepted: int = 0
    n_proposed: int = 0
    history: list[dict] = field(default_factory=list, repr=False)

    @property
    def champion_config(self) -> PipelineConfig:
        return self.champion.config

    @property
    def champion_rules(self) -> RewriteRules | None:
        return self.rulebook.rules_for(self.champion.config)


def optimize_pipeline(
    problem: SearchProblem,
    iterations: int = 40,
    seed: int = 0,
    strategy: str = "evolve",
    use_archive: bool = True,
    warm_starts: list[PipelineConfig] | None = None,
    seed_rules: RewriteRules | None = None,
) -> OptimizeResult:
    """Search pipeline configs for `problem`, scoring every candidate on meta_val.

    strategy="evolve": mutate a parent sampled from the archive (or the champion,
    with use_archive=False). strategy="random": the control arm — independent draws,
    same budget. `warm_starts` (e.g. configs that won on similar problems, see
    transfer.py) are evaluated first and enter the archive if they pass the gate.
    """
    if strategy not in ("evolve", "random"):
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = random.Random(seed)
    rulebook = RuleBook(problem, seed_rules)
    queries = problem.split(OPTIMIZE_SPLIT)
    floor = 0.5 / len(queries)
    seen: dict[PipelineConfig, PipelineScore] = {}

    def evaluate(config: PipelineConfig) -> PipelineScore:
        if config not in seen:
            seen[config] = evaluate_pipeline(problem, config, queries, rulebook.rules_for(config))
        return seen[config]

    baseline = evaluate(BASELINE_PIPELINE)
    archive = Archive(ArchiveEntry(BASELINE_PIPELINE, baseline))
    result = OptimizeResult(problem.name, strategy, archive.entries[0], baseline, archive, rulebook)

    champion = archive.entries[0]
    for config in warm_starts or []:
        entry = ArchiveEntry(config, evaluate(config))
        accepted, crowned = _admit(archive, champion, entry, None, floor, stepping_stones=False)
        champion = entry if crowned else champion
        result.history.append({"iteration": -1, "config": config.label(), "score": seen[config].mean_score,
                               "accepted": accepted, "warm_start": True})

    for i in range(iterations):
        parent = None
        if strategy == "random":
            candidate = random_pipeline_config(rng)
        else:
            parent = _tournament(archive, rng) if use_archive else champion
            candidate = propose_pipeline_mutation(parent.config, rng)
        score = evaluate(candidate)
        entry = ArchiveEntry(candidate, score)
        accepted, crowned = _admit(archive, champion, entry, parent, floor, stepping_stones=use_archive)
        champion = entry if crowned else champion
        result.n_accepted += accepted
        result.history.append({"iteration": i, "config": candidate.label(), "score": score.mean_score,
                               "accepted": accepted})
    result.n_proposed = iterations
    result.champion = champion
    return result
