from __future__ import annotations

from search_rsi.types import Document


def hyde_expand(query: str) -> str:
    """Deterministic stand-in for the generation step in HyDE (Gao et al., 2022,
    "Precise Zero-Shot Dense Retrieval without Relevance Labels"): instead of
    retrieving with the terse query, an LLM first writes a *hypothetical* answer
    passage, and retrieval runs against that passage's richer vocabulary — closing
    the lexical gap between how a question is phrased and how its answer document
    is actually written.

    This function plays the same role RuleBasedPlanner plays for query
    decomposition: a fixed template stands in for the LLM call so the retrieval
    wrapper's contract (query in, expanded pseudo-document out) is testable
    offline. Swapping in a real LLM call is a one-function change — see
    docs/META_RSI.md, generative techniques.
    """
    return (
        f"{query}. This document describes when the term was first established and "
        f"introduced, and how it was later adopted and redefined by a successor "
        f"organization."
    )


class HyDERetriever:
    """Wraps any base retriever, expanding the query before delegating to it.

    Orthogonal to *which* base algorithm runs (BM25/Jaccard/TF-IDF) — the meta-loop
    can toggle HyDE expansion on or off independently of algorithm choice, since in
    principle either helps or hurts depending on how far the query's phrasing is
    from the answer document's phrasing, not on which scoring function is used.
    """

    def __init__(self, base, expander=hyde_expand):
        self.base = base
        self.expander = expander
        self.documents: list[Document] = base.documents

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        return self.base.search(self.expander(query), top_k)
