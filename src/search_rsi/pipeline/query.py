"""Query understanding: spell correction, constraint parsing, temporal intent."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from search_rsi.text import edit_distance, tokenize
from search_rsi.types import Document

_PRICE_MAX = re.compile(r"\b(?:under|below|less than|cheaper than|max)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars|usd)?")
_PRICE_MIN = re.compile(r"\b(?:over|above|more than|at least|min)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars|usd)?")
TEMPORAL_CUES = frozenset({"latest", "newest", "current", "recent", "updated", "today"})
_MAX_FACET_VALUES = 50


@dataclass
class Constraints:
    price_max: float | None = None
    price_min: float | None = None
    facets: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.price_max is not None or self.price_min is not None or bool(self.facets)

    def violations(self, doc: Document) -> int:
        m = doc.metadata
        price = m.get("price")
        n = 0
        if self.price_max is not None and isinstance(price, (int, float)) and price > self.price_max:
            n += 1
        if self.price_min is not None and isinstance(price, (int, float)) and price < self.price_min:
            n += 1
        return n + sum(str(m.get(k, "")).lower() != v for k, v in self.facets.items())


def facet_vocabulary(documents: list[Document]) -> dict[str, set[str]]:
    """Low-cardinality string metadata fields become filterable facets. Derived from
    the corpus alone — never from queries or judgments."""
    values: dict[str, set[str]] = {}
    for doc in documents:
        for key, value in doc.metadata.items():
            if isinstance(value, str):
                values.setdefault(key, set()).add(value.lower())
    return {k: v for k, v in values.items() if len(v) <= _MAX_FACET_VALUES}


def parse_constraints(query: str, facets: dict[str, set[str]]) -> tuple[str, Constraints]:
    """Extract price bounds and exact facet values; returns the query with the
    price phrases removed (the digits in "under $80" are not a text match)."""
    lowered = query.lower()
    constraints = Constraints()
    if m := _PRICE_MAX.search(lowered):
        constraints.price_max = float(m.group(1))
        lowered = lowered.replace(m.group(0), " ")
    if m := _PRICE_MIN.search(lowered):
        constraints.price_min = float(m.group(1))
        lowered = lowered.replace(m.group(0), " ")
    padded = f" {' '.join(tokenize(lowered))} "
    for key, values in facets.items():
        # the longest matching value wins ("running shoes" over "shoes")
        for value in sorted(values, key=len, reverse=True):
            if f" {' '.join(tokenize(value))} " in padded:
                constraints.facets[key] = value
                break
    return lowered, constraints


def has_temporal_intent(query: str) -> bool:
    return bool(TEMPORAL_CUES & set(tokenize(query)))


class SpellCorrector:
    """Replaces out-of-vocabulary tokens with the closest corpus term (edit distance
    1 for short words, 2 for long ones), preferring more frequent terms on ties.
    Tokens a learned rewrite already explains are left alone — correcting the
    user's own vocabulary ("sneakers" -> "speakers") is the classic failure mode."""

    def __init__(self, documents: list[Document]):
        self.df: dict[str, int] = {}
        for doc in documents:
            for tok in set(tokenize(f"{doc.title} {doc.text}")):
                self.df[tok] = self.df.get(tok, 0) + 1
        self._by_len: dict[int, list[str]] = {}
        for tok in self.df:
            self._by_len.setdefault(len(tok), []).append(tok)

    def correct_token(self, token: str) -> str:
        if token in self.df or len(token) < 4 or token.isdigit():
            return token
        limit = 1 if len(token) < 7 else 2
        best: tuple[int, int, str] | None = None
        for n in range(len(token) - limit, len(token) + limit + 1):
            for cand in self._by_len.get(n, ()):
                d = edit_distance(token, cand, limit)
                if d <= limit:
                    key = (d, -self.df[cand], cand)
                    if best is None or key < best:
                        best = key
        return best[2] if best else token

    def correct(self, query: str, protected: set[str] = frozenset()) -> str:
        return " ".join(t if t in protected else self.correct_token(t) for t in tokenize(query))
