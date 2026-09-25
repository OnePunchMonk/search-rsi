from __future__ import annotations

import heapq
import math
from typing import Callable, Sequence

from search_rsi.types import Document

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class DenseIndex:
    """Embedding retrieval with any `embed_fn(texts) -> vectors` (cosine similarity,
    exhaustive scan). The core package stays dependency-free, so no embedder ships by
    default: register one with `search_rsi.retrieval.set_embedder` (for example
    `sentence_transformers_embedder()`), and `dense` becomes a retriever the
    optimizer can select and fuse like any other.
    """

    def __init__(self, documents: list[Document], embed_fn: EmbedFn, texts: list[str] | None = None):
        self.documents = documents
        self.embed_fn = embed_fn
        texts = texts if texts is not None else [d.text for d in documents]
        self._vecs = [_normalize(v) for v in embed_fn(texts)]

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q = _normalize(self.embed_fn([query])[0])
        scores = ((i, sum(a * b for a, b in zip(q, v))) for i, v in enumerate(self._vecs))
        best = heapq.nsmallest(top_k, scores, key=lambda kv: (-kv[1], kv[0]))
        return [(self.documents[i].doc_id, s) for i, s in best if s > 0]


def sentence_transformers_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbedFn:
    """Optional real embedder (`pip install search-rsi[dense]`). Imported lazily so
    the offline path never needs model weights."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed
