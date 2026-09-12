from __future__ import annotations

from typing import Protocol

from search_rsi.retrieval.bm25 import BM25Index
from search_rsi.retrieval.hyde import HyDERetriever
from search_rsi.retrieval.jaccard import JaccardIndex
from search_rsi.retrieval.tfidf import TfidfCosineIndex
from search_rsi.types import Document

RETRIEVER_NAMES = ("bm25", "jaccard", "tfidf")


class Retriever(Protocol):
    documents: list[Document]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...


def build_retriever(
    name: str,
    documents: list[Document],
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    use_hyde: bool = False,
) -> Retriever:
    """The algorithm-selection seam (docs/META_RSI.md): the meta-loop's mutation
    surface includes which retrieval *mechanism* runs, not just this mechanism's
    hyperparameters. Adding a fourth algorithm later (e.g. a real embedding-based
    dense retriever) means adding one branch here and one name to RETRIEVER_NAMES —
    nothing in the harness, evaluator, or meta-loop needs to change.

    `use_hyde` is orthogonal to the base algorithm choice: it wraps whichever base
    retriever was selected with generative query expansion (see retrieval/hyde.py).
    """
    if name == "bm25":
        base: Retriever = BM25Index(documents, k1=bm25_k1, b=bm25_b)
    elif name == "jaccard":
        base = JaccardIndex(documents)
    elif name == "tfidf":
        base = TfidfCosineIndex(documents)
    else:
        raise ValueError(f"unknown retriever: {name!r} (known: {RETRIEVER_NAMES})")
    return HyDERetriever(base) if use_hyde else base
