from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from typing import Callable

from search_rsi.text import tokenize
from search_rsi.types import Document


class TfidfCosineIndex:
    """TF-IDF vectors scored by cosine similarity — a genuinely different scoring
    mechanism from BM25 (no saturation term, no length-normalization knob; instead
    every document vector is unit-normalized, so raw length differences wash out
    entirely rather than being tunably corrected as in BM25's `b`).

    Where this tends to beat BM25: queries that repeat a distinctive multi-word
    term verbatim, since cosine similarity rewards proportional term overlap without
    BM25's diminishing-returns saturation on high-frequency terms. Where it tends to
    lose: very short documents, since a single matching rare term can dominate the
    normalized vector regardless of the rest of the document's relevance.

    With a character-trigram analyzer this same class is the typo-tolerant
    `charngram` retriever (see registry.py).
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
        doc_tokens = [analyzer(t) for t in texts]
        n = len(documents)
        df: Counter[str] = Counter()
        for toks in doc_tokens:
            df.update(set(toks))
        self._idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, toks in enumerate(doc_tokens):
            for term, weight in self._vectorize(toks).items():
                self._postings[term].append((i, weight))

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {term: freq * self._idf.get(term, 0.0) for term, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term, w in self._vectorize(self.analyzer(query)).items():
            for i, dw in self._postings.get(term, ()):
                scores[i] += w * dw
        best = heapq.nsmallest(top_k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self.documents[i].doc_id, s) for i, s in best if s > 0]
