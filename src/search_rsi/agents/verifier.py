from __future__ import annotations


class OverlapVerifier:
    """Checks a candidate citation set actually contains supporting text.

    Rejects citations whose retrieved document text has no lexical overlap with the
    question terms, forcing a re-query rather than letting the answerer commit to
    unsupported evidence (plan.md, Milestone 3).
    """

    def supports(self, question_terms: set[str], doc_text: str) -> bool:
        from search_rsi.retrieval.bm25 import tokenize

        doc_terms = set(tokenize(doc_text))
        return bool(question_terms & doc_terms)
