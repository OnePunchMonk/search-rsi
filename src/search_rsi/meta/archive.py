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
        return max(self.entries, key=lambda e: e.score.mean_score)

    def maybe_accept(self, candidate: ArchiveEntry, min_improvement: float = 0.0) -> bool:
        """Accept if the candidate is not dominated by the current champion on
        score, and either strictly improves score or matches it with fewer tool
        calls (a Pareto acceptance rule, not pure hill-climbing on one number)."""
        champ = self.champion()
        better_score = candidate.score.mean_score > champ.score.mean_score + min_improvement
        tied_score_cheaper = (
            candidate.score.mean_score >= champ.score.mean_score - 1e-9
            and candidate.score.mean_tool_calls < champ.score.mean_tool_calls
        )
        if better_score or tied_score_cheaper:
            self.entries.append(candidate)
            return True
        return False
