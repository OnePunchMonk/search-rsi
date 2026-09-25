from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], k: int = 60, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum 1 / (k + rank).

    Rank-based, so it fuses retrievers whose raw scores live on incomparable scales
    (BM25 vs. cosine) without any score calibration — the standard way hybrid
    lexical + fuzzy (or lexical + dense) search is combined in production.
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(doc_id, len(first_seen))
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return ordered[:top_k] if top_k is not None else ordered
