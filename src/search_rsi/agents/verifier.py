from __future__ import annotations


class OverlapVerifier:
    """Checks a candidate citation set actually contains supporting text.

    Rejects citations whose retrieved document text has fewer than `min_overlap`
    shared terms with the question, forcing a re-query rather than letting the
    answerer commit to unsupported evidence (plan.md, Milestone 3).

    `min_overlap` is one of the parameters the meta-loop (docs/META_RSI.md) is allowed
    to search over: it trades recall (low threshold, more citations accepted) against
    precision (high threshold, fewer false-supported citations).
    """

    def __init__(self, min_overlap: int = 1):
        self.min_overlap = min_overlap

    def supports(self, question_terms: set[str], doc_text: str) -> bool:
        from search_rsi.retrieval.bm25 import tokenize

        doc_terms = set(tokenize(doc_text))
        return len(question_terms & doc_terms) >= self.min_overlap
