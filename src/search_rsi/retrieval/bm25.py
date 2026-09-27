from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from typing import Callable

from search_rsi.text import tokenize  # re-exported: older modules import it from here
from search_rsi.types import Document

__all__ = ["BM25Index", "tokenize"]


class BM25Index:
    """A small, dependency-free BM25 index over real documents.

    No external services, no stubbed scores: term statistics are computed from the
    actual committed corpus text (plan.md C2). Postings are inverted so a query only
    touches documents that share a term with it; `k1`/`b` are applied at query time,
    so one built index serves every hyperparameter setting the optimizer tries.
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
        analyzer: Callable[[str], list[str]] = tokenize,
        texts: list[str] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.analyzer = analyzer
        texts = texts if texts is not None else [d.text for d in documents]
        doc_tokens = [analyzer(t) for t in texts]
        self._doc_len = [len(toks) for toks in doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(doc_tokens):
            for term, freq in Counter(toks).items():
                self._postings[term].append((i, freq))
        n = len(documents)
        self.idf = {
            term: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
            for term, p in self._postings.items()
        }

    def search(
        self, query: str, top_k: int = 5, k1: float | None = None, b: float | None = None
    ) -> list[tuple[str, float]]:
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b
        avgdl = self._avgdl or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in self.analyzer(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, freq in postings:
                denom = freq + k1 * (1 - b + b * self._doc_len[i] / avgdl)
                scores[i] += idf * (freq * (k1 + 1)) / (denom or 1.0)
        best = heapq.nsmallest(top_k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self.documents[i].doc_id, s) for i, s in best if s > 0]
