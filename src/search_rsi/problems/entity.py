"""Navigational entity lookup with misspelled names.

What makes it hard, and which pipeline choices it rewards:
- Queries are names typed with typos ("Kelingtom Ravsworth"): an exact-token
  scorer misses them entirely — character-trigram retrieval or spell
  correction against the corpus vocabulary recovers them.
- Near-duplicate names (Kellington / Kellingham) punish over-eager fuzzy
  matching, so trigram-only is not a free win: fusion with an exact-match
  retriever keeps the precise hits on top.
- Success is binary (did the page the user meant come first?), so the primary
  metric is success@1.
"""
from __future__ import annotations

import random

from search_rsi.problems.base import Query, SearchProblem, round_robin_splits
from search_rsi.types import Document

SYLLABLES = ["ka", "ren", "tor", "vel", "mi", "dra", "lo", "zen", "bar", "thi", "qua", "sol",
             "mor", "len", "gra", "vis", "tal", "rho", "fen", "cu"]
SUFFIXES = ["ington", "ingham", "sworth", "wick", "dale", "mont", "berg", "ford"]
KINDS = [
    ("company", "Industries", "a manufacturer of {thing}", ["turbines", "ceramics", "sensors", "textiles"]),
    ("person", None, "a researcher known for work on {thing}", ["glaciers", "compilers", "vaccines", "optics"]),
    ("place", "Valley", "a region famous for {thing}", ["vineyards", "hot springs", "orchards", "caves"]),
]
CITIES = ["Norhaven", "Belcastle", "Oristad", "Quenmoor", "Talvic", "Draymere"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["drop", "swap", "double", "sub"])
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + rng.choice("aeiourstln") + word[i + 1:]


def build(seed: int = 17) -> SearchProblem:
    rng = random.Random(seed)
    docs: list[Document] = []
    used: set[str] = set()
    stems = []
    while len(stems) < 50:
        stem = "".join(rng.sample(SYLLABLES, 2)).title()
        if stem not in stems:
            stems.append(stem)
    for stem in stems:
        # every stem gets two confusable surnames: the near-duplicate trap
        for suffix in rng.sample(SUFFIXES, 2):
            kind, tail, template, things = KINDS[len(docs) % len(KINDS)]
            first = "".join(rng.sample(SYLLABLES, 2)).title()
            name = f"{first} {stem}{suffix}"
            if tail:
                name = f"{stem}{suffix} {tail}"
            if name in used:
                continue
            used.add(name)
            thing = rng.choice(things)
            year = rng.randint(1890, 2015)
            city = rng.choice(CITIES)
            docs.append(Document(
                doc_id=f"ent{len(docs):03d}",
                title=name,
                text=f"{name} is {template.format(thing=thing)}, based in {city}. Established {year}.",
                metadata={"kind": kind, "city": city, "year": year},
            ))

    queries: list[Query] = []
    for doc in rng.sample(docs, 90):
        words = doc.title.split()
        k = len(queries) % 3
        if k == 0:
            typed = [_typo(w, rng) for w in words]
        elif k == 1:
            typed = [_typo(w, rng) if i == len(words) - 1 or len(words) == 1 else w for i, w in enumerate(words)]
        else:
            typed = [_typo(w, rng) for w in words] + [rng.choice(["founded", "location", "history"])]
        queries.append(Query(qid=f"eq{len(queries):03d}", text=" ".join(typed).lower(),
                             qrels={doc.doc_id: 2}, intent=doc.metadata["kind"]))

    problem = SearchProblem(
        name="entity_lookup",
        use_case="navigational name search with typos and near-duplicate names",
        description=__doc__.strip().splitlines()[0],
        documents=docs,
        queries=queries,
        splits=round_robin_splits(queries, group=lambda q: q.intent),
        primary_metric="success@1",
    )
    problem.validate()
    return problem
