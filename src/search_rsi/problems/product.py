"""E-commerce product search.

What makes it hard, and which pipeline choices it rewards:
- Users say "sneakers", the catalog says "running shoes": a vocabulary gap no
  lexical scorer closes on its own — the learned rewrite memory has to.
- Queries carry hard constraints ("under $80", a color, a brand) that only a
  structured filter enforces; lexical match on the digits "80" is meaningless.
- Singular/plural mismatch ("running shoe" vs. "shoes") rewards stemming.
"""
from __future__ import annotations

import random

from search_rsi.problems.base import Query, SearchProblem, round_robin_splits
from search_rsi.types import Document

CATEGORIES = {
    # catalog term: (user synonyms, price range, materials, uses)
    "running shoes": (["sneakers", "trainers"], (40, 160), ["mesh", "knit"], "road running and daily training"),
    "hiking boots": (["trekking footwear", "mountaineering footwear"], (80, 260), ["leather", "suede"], "rough trails and wet terrain"),
    "rain jacket": (["raincoat", "slicker"], (50, 220), ["nylon", "polyester"], "staying dry in heavy rain"),
    "backpack": (["rucksack", "daypack"], (30, 180), ["canvas", "nylon"], "commuting and weekend trips"),
    "headphones": (["earbuds", "earphones"], (20, 300), ["plastic", "aluminum"], "music, calls and travel"),
    "water bottle": (["flask", "canteen"], (10, 50), ["steel", "plastic"], "hydration on the go"),
    "yoga mat": (["exercise pad", "workout pad"], (15, 90), ["rubber", "cork"], "yoga, pilates and stretching"),
    "desk lamp": (["reading light", "task light"], (20, 120), ["aluminum", "steel"], "focused light at a desk"),
}
BRANDS = ["Altura", "Brisk", "Corvid", "Dunmore", "Evora", "Fjell"]
COLORS = ["red", "blue", "black", "green", "grey", "white"]
MODELS = ["One", "Pro", "Lite", "Max", "Trail", "Air", "Core", "Flex"]
FEATURES = ["lightweight build", "two year warranty", "recycled materials", "reinforced seams", "compact design", "easy care"]


def _singular(category: str) -> str:
    return category[:-1] if category.endswith("s") else category


def _grade(doc: Document, spec: dict) -> int:
    m = doc.metadata
    if m["category"] != spec["category"]:
        return 0
    checks = []
    if "color" in spec:
        checks.append(m["color"] == spec["color"])
    if "brand" in spec:
        checks.append(m["brand"] == spec["brand"])
    if "price_max" in spec:
        checks.append(m["price"] <= spec["price_max"])
    misses = checks.count(False)
    return 2 if misses == 0 else (1 if misses == 1 and len(checks) > 1 else 0)


def build(seed: int = 7) -> SearchProblem:
    rng = random.Random(seed)
    docs: list[Document] = []
    for cat, (_, (lo, hi), materials, use) in CATEGORIES.items():
        for brand in BRANDS:
            for _ in range(3):
                color = rng.choice(COLORS)
                model = f"{rng.choice(MODELS)} {rng.randint(1, 9)}"
                price = rng.randrange(lo, hi, 5)
                material = rng.choice(materials)
                feats = rng.sample(FEATURES, 2)
                doc_id = f"p{len(docs):03d}"
                docs.append(Document(
                    doc_id=doc_id,
                    title=f"{brand} {model} {cat.title()} - {color.title()}",
                    text=(f"The {brand} {model} is a {color} {material} {cat if cat == 'headphones' else _singular(cat)} "
                          f"built for {use}. Features: {feats[0]}, {feats[1]}. Price ${price}."),
                    metadata={"category": cat, "brand": brand, "color": color, "price": price},
                ))

    queries: list[Query] = []
    for cat, (synonyms, (lo, hi), _, _) in CATEGORIES.items():
        in_cat = [d for d in docs if d.metadata["category"] == cat]
        terms = [cat, cat, _singular(cat)] + synonyms * 2
        for j in range(12):
            anchor = rng.choice(in_cat).metadata  # guarantees at least one exact match
            term = terms[j % len(terms)]
            kind = j % 3
            if kind == 0:
                budget = anchor["price"] + rng.choice([0, 5, 10, 20])
                spec = {"category": cat, "color": anchor["color"], "price_max": budget}
                text = f"{anchor['color']} {term} under ${budget}"
            elif kind == 1:
                spec = {"category": cat, "brand": anchor["brand"], "color": anchor["color"]}
                text = f"{anchor['brand']} {term} in {anchor['color']}"
            else:
                budget = anchor["price"] + rng.choice([0, 5, 15])
                spec = {"category": cat, "price_max": budget}
                text = f"{term} below {budget} dollars"
            qrels = {d.doc_id: g for d in docs if (g := _grade(d, spec)) > 0}
            queries.append(Query(qid=f"pq{len(queries):03d}", text=text, qrels=qrels, intent=term))

    problem = SearchProblem(
        name="product_search",
        use_case="e-commerce catalog search with synonyms and hard constraints",
        description=__doc__.strip().splitlines()[0],
        documents=docs,
        queries=queries,
        splits=round_robin_splits(queries, group=lambda q: q.intent),
        primary_metric="ndcg@10",
    )
    problem.validate()
    return problem
