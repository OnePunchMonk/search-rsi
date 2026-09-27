from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from search_rsi.memory.rewrites import RewriteRules
from search_rsi.pipeline.config import PipelineConfig
from search_rsi.pipeline.query import (
    Constraints,
    SpellCorrector,
    facet_vocabulary,
    has_temporal_intent,
    parse_constraints,
)
from search_rsi.retrieval import build_retriever, reciprocal_rank_fusion
from search_rsi.retrieval.bm25 import BM25Index
from search_rsi.text import STOPWORDS, get_analyzer, tokenize
from search_rsi.types import Document

_PRF_DOCS = 3
_RERANK_POOL = 30


@dataclass
class SearchResponse:
    hits: list[tuple[str, float]]
    query: str  # the query as finally executed, after every rewrite
    constraints: Constraints = field(default_factory=Constraints)
    retrieval_calls: int = 0

    @property
    def doc_ids(self) -> list[str]:
        return [d for d, _ in self.hits]


class SearchPipeline:
    """rewrite -> correct -> parse constraints -> retrieve (+fuse) -> PRF ->
    filter -> rerank -> recency, each stage switched and tuned by PipelineConfig.

    `cache` lets many configs over the same corpus share built indexes; pass a
    problem's `index_cache` (the optimizer does) or leave it None for a standalone
    pipeline.
    """

    def __init__(
        self,
        documents: list[Document],
        config: PipelineConfig,
        rewrites: RewriteRules | None = None,
        cache: dict | None = None,
    ):
        self.documents = documents
        self.config = config
        self.rewrites = rewrites or RewriteRules()
        self._cache = cache if cache is not None else {}
        self._by_id = self._cached(("by_id",), lambda: {d.doc_id: d for d in documents})
        texts = self._cached(("texts", config.title_boost),
                             lambda: [d.indexed_text(config.title_boost) for d in documents])
        self._primary = self._index(config.retriever, texts)
        self._secondary = (
            self._index(config.fusion_retriever, texts)
            if config.fusion_retriever not in ("none", config.retriever)
            else None
        )
        self._analyze = get_analyzer(config.analyzer)
        if config.spell_correct:
            self._speller = self._cached(("speller",), lambda: SpellCorrector(documents))
        if config.filter_mode != "none":
            self._facets = self._cached(("facets",), lambda: facet_vocabulary(documents))
        if config.prf_terms:
            self._prf_stats = self._cached(("prf", config.analyzer, config.title_boost),
                                           lambda: BM25Index(documents, analyzer=self._analyze, texts=texts))

    def _cached(self, key: tuple, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def _index(self, name: str, texts: list[str]):
        # charngram and dense ignore the word analyzer, so they share one index
        analyzer = self.config.analyzer if name in ("bm25", "tfidf", "jaccard") else None
        key = ("index", name, analyzer, self.config.title_boost)
        return self._cached(key, lambda: build_retriever(
            name, self.documents, analyzer=self.config.analyzer, texts=texts))

    def _retrieve(self, query: str, depth: int) -> tuple[list[tuple[str, float]], int]:
        c = self.config
        if isinstance(self._primary, BM25Index):
            primary = self._primary.search(query, depth, k1=c.bm25_k1, b=c.bm25_b)
        else:
            primary = self._primary.search(query, depth)
        if self._secondary is None:
            return primary, 1
        if isinstance(self._secondary, BM25Index):
            secondary = self._secondary.search(query, depth, k1=c.bm25_k1, b=c.bm25_b)
        else:
            secondary = self._secondary.search(query, depth)
        return reciprocal_rank_fusion([primary, secondary], k=c.rrf_k, top_k=depth), 2

    def _prf_expand(self, query: str, hits: list[tuple[str, float]]) -> str:
        """RM3-lite: add the highest tf-idf terms of the top feedback documents."""
        stats: BM25Index = self._prf_stats
        q_terms = set(self._analyze(query))
        weights: Counter[str] = Counter()
        for doc_id, _ in hits[:_PRF_DOCS]:
            doc = self._by_id[doc_id]
            for term, tf in Counter(self._analyze(doc.indexed_text(self.config.title_boost))).items():
                if term not in q_terms and term not in STOPWORDS and not term.isdigit():
                    weights[term] += tf * stats.idf.get(term, 0.0)
        extra = [t for t, _ in weights.most_common(self.config.prf_terms)]
        return f"{query} {' '.join(extra)}" if extra else query

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        c = self.config
        original = query
        protected: set[str] = set()
        if c.use_learned_rewrites and len(self.rewrites):
            protected = {t for t in tokenize(query) if self.rewrites.get(t)}
            query = self.rewrites.apply(query)
        if c.spell_correct:
            query = self._speller.correct(query, protected)
        constraints = Constraints()
        if c.filter_mode != "none":
            query, constraints = parse_constraints(query, self._facets)

        hits, calls = self._retrieve(query, c.depth)
        if c.prf_terms and hits:
            query = self._prf_expand(query, hits)
            hits, more = self._retrieve(query, c.depth)
            calls += more
        if constraints:
            violations = {d: constraints.violations(self._by_id[d]) for d, _ in hits}
            if c.filter_mode == "hard":
                hits = [(d, s) for d, s in hits if violations[d] == 0]
            else:  # soft: stable sort by number of violated constraints
                hits = sorted(hits, key=lambda h: violations[h[0]])

        if hits and (c.rerank_weight > 0 or (c.recency_weight > 0 and has_temporal_intent(original))):
            hits = self._rescore(original, hits, violations if constraints else None)
        return SearchResponse(hits=hits[:top_k], query=query, constraints=constraints, retrieval_calls=calls)

    def _rescore(
        self, original: str, hits: list[tuple[str, float]], violations: dict[str, int] | None
    ) -> list[tuple[str, float]]:
        c = self.config
        pool = hits[:_RERANK_POOL]
        top = pool[0][1] or 1.0
        rescored = {d: s / top for d, s in pool}
        if c.rerank_weight > 0:
            q_terms = set(self._analyze(original)) - STOPWORDS
            if q_terms:
                for d in rescored:
                    doc_terms = set(self._analyze(self._by_id[d].indexed_text()))
                    coverage = len(q_terms & doc_terms) / len(q_terms)
                    # penalize documents padded with terms the query never asked for,
                    # so the exact article beats its longer near-duplicate on ties
                    precision = len(q_terms & doc_terms) / math.sqrt(len(doc_terms) or 1)
                    rescored[d] = (1 - c.rerank_weight) * rescored[d] + c.rerank_weight * (coverage + 0.1 * precision)
        if c.recency_weight > 0 and has_temporal_intent(original):
            dates = {d: self._by_id[d].metadata.get("date") for d in rescored}
            known = [v for v in dates.values() if isinstance(v, (int, float))]
            if len(known) > 1 and max(known) > min(known):
                lo, hi = min(known), max(known)
                for d, v in dates.items():
                    recency = (v - lo) / (hi - lo) if isinstance(v, (int, float)) else 0.0
                    rescored[d] = (1 - c.recency_weight) * rescored[d] + c.recency_weight * recency
        order = {d: i for i, (d, _) in enumerate(hits)}
        # rescoring never overrides the constraint ordering filters established
        v = violations or {}
        head = sorted(rescored.items(), key=lambda kv: (v.get(kv[0], 0), -kv[1], order[kv[0]]))
        return head + hits[_RERANK_POOL:]
