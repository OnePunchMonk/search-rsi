from __future__ import annotations

import random
from dataclasses import dataclass

from search_rsi.meta.config import HarnessConfig
from search_rsi.meta.evaluate import VariantScore


@dataclass(frozen=True)
class ArchiveEntry:
    config: HarnessConfig
    score: VariantScore


class Archive:
    """A population of accepted harness variants, not a single 'best so far'.

    DGM's finding that motivates this: always mutating the current champion gets
    stuck in local optima that a stepping-stone variant (worse on its own, but a
    better parent for the next mutation) could have escaped. Sampling the parent
    uniformly from the archive, rather than always taking the champion, is the whole
    mechanism — everything else is bookkeeping.
    """

    def __init__(self, seed_entry: ArchiveEntry):
        self.entries: list[ArchiveEntry] = [seed_entry]

    def sample_parent(self, rng: random.Random) -> HarnessConfig:
        return rng.choice(self.entries).config

    def champion(self) -> ArchiveEntry:
        """Highest score; ties broken by lower latency, since the product goal is
        best-scoring *and* lowest-latency, not score alone (see pareto_front for the
        full tradeoff surface rather than a single number)."""
        return max(self.entries, key=lambda e: (e.score.mean_score, -e.score.mean_latency_ms))

    def pareto_front(self) -> list[ArchiveEntry]:
        """Entries not dominated on (higher score, lower latency) by any other
        entry — the actual deliverable for 'describe a search problem, get the
        best-scoring lowest-latency config': a shortlist of real tradeoffs
        (a slower-but-more-accurate option and a faster-but-slightly-worse option),
        not a single collapsed number that hides the choice from whoever deploys it.
        """
        front = []
        for candidate in self.entries:
            dominated = any(
                other.score.mean_score >= candidate.score.mean_score
                and other.score.mean_latency_ms <= candidate.score.mean_latency_ms
                and other is not candidate
                and (
                    other.score.mean_score > candidate.score.mean_score
                    or other.score.mean_latency_ms < candidate.score.mean_latency_ms
                )
                for other in self.entries
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda e: -e.score.mean_score)

    def maybe_accept(self, candidate: ArchiveEntry, min_improvement: float) -> bool:
        """Accept if the candidate is not dominated by the current champion on
        score, and either improves score by more than `min_improvement` or matches
        it with fewer tool calls or lower latency (a Pareto acceptance rule, not
        pure hill-climbing on one number).

        `min_improvement` is a significance floor, not decoration: on a small
        meta-val split, a single task flipping from wrong to right can look like a
        real win when it's within noise of re-running the same config. Callers
        should size it relative to what one task's worth of score swing means for
        their split (docs/META_RSI.md "Current status" — this was 0.0 in the v1
        slice, which is why a single lucky task could get accepted).
        """
        champ = self.champion()
        better_score = candidate.score.mean_score > champ.score.mean_score + min_improvement
        tied_score_cheaper = (
            candidate.score.mean_score >= champ.score.mean_score - 1e-9
            and (
                candidate.score.mean_tool_calls < champ.score.mean_tool_calls
                or candidate.score.mean_latency_ms < champ.score.mean_latency_ms * 0.9
            )
        )
        if better_score or tied_score_cheaper:
            self.entries.append(candidate)
            return True
        return False
