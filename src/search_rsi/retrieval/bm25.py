from __future__ import annotations

import math
import re
from collections import Counter

from search_rsi.types import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A small, dependency-free BM25 index over real documents.

    No external services, no stubbed scores: term statistics are computed from the
    actual committed corpus text (plan.md C2).
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self._doc_tokens = [tokenize(d.text) for d in documents]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        self._term_freqs = [Counter(toks) for toks in self._doc_tokens]
        self._df: Counter[str] = Counter()
        for tf in self._term_freqs:
            for term in tf:
                self._df[term] += 1
        n = len(documents)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, doc in enumerate(self.documents):
            tf = self._term_freqs[i]
            dl = self._doc_len[i]
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
            if score > 0:
                scores.append((doc.doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
