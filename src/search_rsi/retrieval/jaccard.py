from __future__ import annotations

from search_rsi.retrieval.bm25 import tokenize
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

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._doc_terms = [set(tokenize(d.text)) for d in documents]

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_terms = set(tokenize(query))
        scores: list[tuple[str, float]] = []
        for doc, terms in zip(self.documents, self._doc_terms):
            union = q_terms | terms
            if not union:
                continue
            score = len(q_terms & terms) / len(union)
            if score > 0:
                scores.append((doc.doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
