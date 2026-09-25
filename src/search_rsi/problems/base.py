from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

from search_rsi.types import Document

SPLIT_NAMES = ("train", "meta_val", "held_out_eval")


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    qrels: dict = field(hash=False)  # doc_id -> graded relevance (0 = not relevant)
    intent: str = "general"


@dataclass
class SearchProblem:
    """One search use case: a corpus, graded queries, and a frozen 3-way split.

    `train` feeds the inner loop (learned query-rewrite memory), `meta_val` is the
    only split the pipeline optimizer's accept/reject gate may see, and
    `held_out_eval` is scored once at the end to report whether a win generalizes
    (docs/META_RSI.md, Guardrail 2). `primary_metric` is the single executed score
    that drives decisions for this use case (plan.md C1).
    """

    name: str
    use_case: str
    description: str
    documents: list[Document]
    queries: list[Query]
    splits: dict[str, list[str]]
    primary_metric: str = "ndcg@10"
    # Retrieval indexes are expensive relative to a query; configs that share an
    # index key (retriever, analyzer, title boost) share one built index.
    index_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def split(self, name: str) -> list[Query]:
        by_id = {q.qid: q for q in self.queries}
        return [by_id[qid] for qid in self.splits[name]]

    def validate(self) -> None:
        doc_ids = {d.doc_id for d in self.documents}
        if len(doc_ids) != len(self.documents):
            raise ValueError(f"{self.name}: duplicate doc ids")
        qids = [q.qid for q in self.queries]
        if len(set(qids)) != len(qids):
            raise ValueError(f"{self.name}: duplicate query ids")
        for q in self.queries:
            missing = set(q.qrels) - doc_ids
            if missing:
                raise ValueError(f"{self.name}: query {q.qid} judges unknown docs {sorted(missing)[:3]}")
            if not any(g > 0 for g in q.qrels.values()):
                raise ValueError(f"{self.name}: query {q.qid} has no relevant document")
        seen: set[str] = set()
        for name in SPLIT_NAMES:
            ids = set(self.splits.get(name, ()))
            if ids & seen:
                raise ValueError(f"{self.name}: split {name} overlaps another split")
            seen |= ids
        if not seen <= set(qids):
            raise ValueError(f"{self.name}: split references unknown query ids")

    def summary(self) -> dict:
        return {
            "name": self.name,
            "use_case": self.use_case,
            "documents": len(self.documents),
            "queries": len(self.queries),
            **{name: len(self.splits.get(name, ())) for name in SPLIT_NAMES},
            "primary_metric": self.primary_metric,
        }


def round_robin_splits(queries: Iterable[Query], group: Callable[[Query], str]) -> dict[str, list[str]]:
    """Deal each group's queries across train/meta_val/held_out_eval in turn, so
    every split sees every query pattern (every synonym, every intent) — otherwise
    'did the win generalize' would be confounded with 'did the split happen to
    contain that pattern at all'."""
    counters: dict[str, int] = {}
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for q in queries:
        key = group(q)
        if key not in counters:
            # stagger each group's starting split so small groups don't all
            # land their first (often only) query in train
            counters[key] = len(counters)
        i = counters[key]
        counters[key] = i + 1
        splits[SPLIT_NAMES[i % 3]].append(q.qid)
    return splits


def hash_splits(qids: Iterable[str], fractions: tuple[float, float] = (0.4, 0.3)) -> dict[str, list[str]]:
    """Deterministic content-hash split for imported datasets: stable across runs
    and machines, independent of file order."""
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for qid in qids:
        bucket = int(hashlib.sha1(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < fractions[0]:
            splits["train"].append(qid)
        elif bucket < fractions[0] + fractions[1]:
            splits["meta_val"].append(qid)
        else:
            splits["held_out_eval"].append(qid)
    return splits
