from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """The full mutable surface of the search-pipeline optimizer.

    Every stage a production search stack tunes per use case is a field here: text
    analysis, the retrieval mechanism, hybrid fusion, field weighting, query
    rewriting (spell correction, pseudo-relevance feedback, learned rewrites),
    structured filtering, reranking and a recency prior. There is no field for the
    grader, the qrels, or the split boundaries, so no mutation can reach them
    (docs/META_RSI.md, Guardrail 1).
    """

    analyzer: str = "plain"  # plain | stem | stem_stop | code
    retriever: str = "bm25"  # bm25 | tfidf | jaccard | charngram | dense
    fusion_retriever: str = "none"  # second retriever fused by RRF, or none
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    title_boost: int = 0  # extra copies of the title field in the indexed text
    spell_correct: bool = False
    prf_terms: int = 0  # pseudo-relevance-feedback expansion terms (0 = off)
    filter_mode: str = "none"  # none | soft (demote violations) | hard (drop them)
    recency_weight: float = 0.0  # recency prior, gated on temporal intent in the query
    rerank_weight: float = 0.0  # query-term-coverage rerank of the top candidates
    use_learned_rewrites: bool = False  # apply rules from the rewrite memory
    depth: int = 100  # candidates retrieved before filtering / reranking

    def label(self) -> str:
        default = PipelineConfig()
        parts = [f"{self.retriever}"]
        if self.fusion_retriever not in ("none", self.retriever):
            parts[0] += f"+{self.fusion_retriever}"
        for f in dataclasses.fields(self):
            if f.name in ("retriever", "fusion_retriever"):
                continue
            value = getattr(self, f.name)
            if value != getattr(default, f.name):
                if f.name.startswith("bm25_") and "bm25" not in (self.retriever, self.fusion_retriever):
                    continue
                if f.name == "rrf_k" and self.fusion_retriever in ("none", self.retriever):
                    continue
                shown = f"{value:.2f}" if isinstance(value, float) else str(value)
                parts.append(f"{f.name}={shown}")
        return parts[0] + (f"[{','.join(parts[1:])}]" if len(parts) > 1 else "")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


BASELINE_PIPELINE = PipelineConfig()
