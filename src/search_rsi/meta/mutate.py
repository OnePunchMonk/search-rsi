from __future__ import annotations

import random
from dataclasses import replace

from search_rsi.meta.config import HarnessConfig
from search_rsi.retrieval import RETRIEVER_NAMES

_MUTABLE_FIELDS = ["retriever", "bm25_k1", "bm25_b", "verifier_min_overlap", "use_hyde"]


def propose_mutation(parent: HarnessConfig, rng: random.Random) -> HarnessConfig:
    """Perturb one field of the parent config (coordinate-wise, DGM/AlphaEvolve-style
    small local edits rather than resampling the whole config at once).

    Mutating `retriever` swaps the whole scoring mechanism (BM25 vs. Jaccard vs.
    TF-IDF cosine) — this is the algorithm-selection dimension: the meta-loop isn't
    just tuning one algorithm's knobs, it can decide a different algorithm altogether
    fits this benchmark's task shape better. `bm25_k1`/`bm25_b` mutations only matter
    when `retriever == "bm25"`; evaluate_variant simply ignores them otherwise.

    This is also the seam that later grows into an LLM-proposed *code* diff (new
    retrieval algorithm implementation, not just a choice among existing ones) — see
    docs/META_RSI.md, Milestone 2. The accept/reject and archive machinery around it
    does not need to change when that happens.
    """
    field = rng.choice(_MUTABLE_FIELDS)
    if field == "retriever":
        other_names = [n for n in RETRIEVER_NAMES if n != parent.retriever]
        return replace(parent, retriever=rng.choice(other_names))
    if field == "bm25_k1":
        return replace(parent, bm25_k1=max(0.1, parent.bm25_k1 + rng.uniform(-0.5, 0.5)))
    if field == "bm25_b":
        return replace(parent, bm25_b=min(1.0, max(0.0, parent.bm25_b + rng.uniform(-0.2, 0.2))))
    if field == "use_hyde":
        return replace(parent, use_hyde=not parent.use_hyde)
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
        retriever=rng.choice(RETRIEVER_NAMES),
        bm25_k1=rng.uniform(0.5, 3.0),
        bm25_b=rng.uniform(0.0, 1.0),
        verifier_min_overlap=rng.choice([1, 2, 3]),
        use_hyde=rng.choice([True, False]),
    )
