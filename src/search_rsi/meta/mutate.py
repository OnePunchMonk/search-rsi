from __future__ import annotations

import random
from dataclasses import replace

from search_rsi.meta.config import HarnessConfig


def propose_mutation(parent: HarnessConfig, rng: random.Random) -> HarnessConfig:
    """Perturb one field of the parent config (coordinate-wise, DGM/AlphaEvolve-style
    small local edits rather than resampling the whole config at once).

    This is the seam that later grows into an LLM-proposed *code* diff (planner
    logic, retrieval algorithm choice) rather than a numeric perturbation — see
    docs/META_RSI.md, Milestone 2. The accept/reject and archive machinery around it
    does not need to change when that happens.
    """
    field = rng.choice(["bm25_k1", "bm25_b", "verifier_min_overlap"])
    if field == "bm25_k1":
        return replace(parent, bm25_k1=max(0.1, parent.bm25_k1 + rng.uniform(-0.5, 0.5)))
    if field == "bm25_b":
        return replace(parent, bm25_b=min(1.0, max(0.0, parent.bm25_b + rng.uniform(-0.2, 0.2))))
    return replace(
        parent,
        verifier_min_overlap=max(1, parent.verifier_min_overlap + rng.choice([-1, 1])),
    )


def propose_random_config(rng: random.Random) -> HarnessConfig:
    """The control arm: a config drawn independently of any parent or prior
    evaluation, used to distinguish 'the search found something' from 'this region
    of config space is just generally fine' (docs/META_RSI.md, Guardrail: meta
    ablation)."""
    return HarnessConfig(
        bm25_k1=rng.uniform(0.5, 3.0),
        bm25_b=rng.uniform(0.0, 1.0),
        verifier_min_overlap=rng.choice([1, 2, 3]),
    )
