from __future__ import annotations

import random
from dataclasses import dataclass

from search_rsi.meta.archive import Archive, ArchiveEntry
from search_rsi.meta.config import BASELINE, HarnessConfig
from search_rsi.meta.evaluate import evaluate_variant
from search_rsi.meta.mutate import propose_mutation, propose_random_config
from search_rsi.types import Task


@dataclass(frozen=True)
class MetaLoopResult:
    champion_config: HarnessConfig
    champion_score: float
    baseline_score: float
    n_accepted: int
    n_proposed: int


def run_meta_loop(
    meta_val_tasks: list[Task],
    iterations: int,
    seed: int = 0,
    use_archive: bool = True,
) -> MetaLoopResult:
    """The meta-loop: propose a config mutation, evaluate it against the frozen
    meta-val split (never train, never held-out eval), accept into the archive if
    it's not dominated by the current champion.

    `use_archive=False` reproduces plain hill-climbing (always mutate the champion)
    for comparison against the archive-based DGM-style search — see
    experiments/run_meta_ablation.py for the full control arm (evolved vs. random).
    """
    rng = random.Random(seed)
    baseline_score = evaluate_variant(BASELINE, meta_val_tasks).mean_score
    archive = Archive(ArchiveEntry(BASELINE, evaluate_variant(BASELINE, meta_val_tasks)))

    n_accepted = 0
    for _ in range(iterations):
        parent = archive.sample_parent(rng) if use_archive else archive.champion().config
        candidate_config = propose_mutation(parent, rng)
        candidate_score = evaluate_variant(candidate_config, meta_val_tasks)
        if archive.maybe_accept(ArchiveEntry(candidate_config, candidate_score)):
            n_accepted += 1

    champ = archive.champion()
    return MetaLoopResult(
        champion_config=champ.config,
        champion_score=champ.score.mean_score,
        baseline_score=baseline_score,
        n_accepted=n_accepted,
        n_proposed=iterations,
    )


def run_random_search_control(
    meta_val_tasks: list[Task], iterations: int, seed: int = 0
) -> MetaLoopResult:
    """The meta-level ablation arm: same evaluation budget, but each candidate is an
    independently sampled config rather than a mutation of an archived parent. If
    the evolved arm doesn't beat this, the search isn't doing anything the config
    space wasn't already going to hand you for free."""
    rng = random.Random(seed)
    baseline_score = evaluate_variant(BASELINE, meta_val_tasks).mean_score
    archive = Archive(ArchiveEntry(BASELINE, evaluate_variant(BASELINE, meta_val_tasks)))

    n_accepted = 0
    for _ in range(iterations):
        candidate_config = propose_random_config(rng)
        candidate_score = evaluate_variant(candidate_config, meta_val_tasks)
        if archive.maybe_accept(ArchiveEntry(candidate_config, candidate_score)):
            n_accepted += 1

    champ = archive.champion()
    return MetaLoopResult(
        champion_config=champ.config,
        champion_score=champ.score.mean_score,
        baseline_score=baseline_score,
        n_accepted=n_accepted,
        n_proposed=iterations,
    )
