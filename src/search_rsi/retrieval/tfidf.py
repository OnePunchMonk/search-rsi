from __future__ import annotations

import math
from collections import Counter

from search_rsi.retrieval.bm25 import tokenize
from search_rsi.types import Document


class TfidfCosineIndex:
    """TF-IDF vectors scored by cosine similarity — a third, genuinely different
    scoring mechanism from BM25 (no saturation term, no length-normalization knob;
    instead every document vector is unit-normalized, so raw length differences
    wash out entirely rather than being tunably corrected as in BM25's `b`).

    Where this tends to beat BM25: queries that repeat a distinctive multi-word
    term verbatim, since cosine similarity rewards proportional term overlap without
    BM25's diminishing-returns saturation on high-frequency terms. Where it tends to
    lose: very short documents, since a single matching rare term can dominate the
    normalized vector regardless of the rest of the document's relevance.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents
        doc_tokens = [tokenize(d.text) for d in documents]
        n = len(documents)
        df: Counter[str] = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1
        self._idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
        self._doc_vecs = [self._vectorize(toks) for toks in doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {term: freq * self._idf.get(term, 0.0) for term, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_vec = self._vectorize(tokenize(query))
        scores: list[tuple[str, float]] = []
        for doc, doc_vec in zip(self.documents, self._doc_vecs):
            score = sum(w * doc_vec.get(term, 0.0) for term, w in q_vec.items())
            if score > 0:
                scores.append((doc.doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
