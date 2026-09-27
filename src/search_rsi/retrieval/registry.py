from __future__ import annotations

from typing import Protocol

from search_rsi.retrieval.bm25 import BM25Index
from search_rsi.retrieval.dense import DenseIndex, EmbedFn
from search_rsi.retrieval.hyde import HyDERetriever
from search_rsi.retrieval.jaccard import JaccardIndex
from search_rsi.retrieval.tfidf import TfidfCosineIndex
from search_rsi.text import TRIGRAM_ANALYZER, get_analyzer
from search_rsi.types import Document

# The agentic Harness's meta-loop (search_rsi.meta) selects among these three.
RETRIEVER_NAMES = ("bm25", "jaccard", "tfidf")

_embedder: EmbedFn | None = None


def set_embedder(embed_fn: EmbedFn | None) -> None:
    """Register (or clear, with None) the embedding function behind `dense`."""
    global _embedder
    _embedder = embed_fn


def pipeline_retriever_names() -> tuple[str, ...]:
    """Retrievers the search-pipeline optimizer may select: every lexical mechanism,
    the typo-tolerant character-trigram index, and `dense` once an embedder is set."""
    names = ("bm25", "tfidf", "jaccard", "charngram")
    return names + ("dense",) if _embedder is not None else names


class Retriever(Protocol):
    documents: list[Document]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...


def build_retriever(
    name: str,
    documents: list[Document],
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    use_hyde: bool = False,
    analyzer: str = "plain",
    texts: list[str] | None = None,
) -> Retriever:
    """The algorithm-selection seam (docs/META_RSI.md): the meta-loop's mutation
    surface includes which retrieval *mechanism* runs, not just this mechanism's
    hyperparameters. Adding an algorithm means adding one branch here — nothing in
    the harness, evaluator, or meta-loop needs to change.

    `use_hyde` is orthogonal to the base algorithm choice: it wraps whichever base
    retriever was selected with generative query expansion (see retrieval/hyde.py).
    `texts` overrides what gets indexed per document (e.g. title-boosted text).
    """
    analyze = get_analyzer(analyzer)
    if name == "bm25":
        base: Retriever = BM25Index(documents, k1=bm25_k1, b=bm25_b, analyzer=analyze, texts=texts)
    elif name == "jaccard":
        base = JaccardIndex(documents, analyzer=analyze, texts=texts)
    elif name == "tfidf":
        base = TfidfCosineIndex(documents, analyzer=analyze, texts=texts)
    elif name == "charngram":
        base = TfidfCosineIndex(documents, analyzer=TRIGRAM_ANALYZER, texts=texts)
    elif name == "dense":
        if _embedder is None:
            raise ValueError("dense retriever needs an embedder: call search_rsi.retrieval.set_embedder(fn)")
        base = DenseIndex(documents, _embedder, texts=texts)
    else:
        raise ValueError(f"unknown retriever: {name!r} (known: {pipeline_retriever_names()})")
    return HyDERetriever(base) if use_hyde else base
