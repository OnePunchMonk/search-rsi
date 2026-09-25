from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Callable

from search_rsi.text import tokenize
from search_rsi.types import Document


class JaccardIndex:
    """Set-overlap retrieval: score = |query terms ∩ doc terms| / |query terms ∪ doc terms|.

    Deliberately not a BM25 variant: no term frequency weighting, no IDF, no length
    normalization. It is strong when documents are short and roughly uniform in
    length (no length bias to correct for) and weak when a long document that
    genuinely covers the query gets penalized by the union term for containing many
    unrelated words. Existing on the corpus alongside BM25 is what makes "which
    algorithm wins on this task shape" an actual, checkable question rather than an
    assumption (docs/META_RSI.md).
    """

    def __init__(
        self,
        documents: list[Document],
        analyzer: Callable[[str], list[str]] = tokenize,
        texts: list[str] | None = None,
    ):
        self.documents = documents
        self.analyzer = analyzer
        texts = texts if texts is not None else [d.text for d in documents]
        self._doc_sizes: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        for i, text in enumerate(texts):
            terms = set(analyzer(text))
            self._doc_sizes.append(len(terms))
            for term in terms:
                self._postings[term].append(i)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_terms = set(self.analyzer(query))
        overlap: dict[int, int] = defaultdict(int)
        for term in q_terms:
            for i in self._postings.get(term, ()):
                overlap[i] += 1
        scores = {i: n / (len(q_terms) + self._doc_sizes[i] - n) for i, n in overlap.items()}
        best = heapq.nsmallest(top_k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self.documents[i].doc_id, s) for i, s in best if s > 0]
