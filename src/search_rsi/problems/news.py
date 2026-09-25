"""Time-sensitive search over a report archive with yearly editions.

What makes it hard, and which pipeline choices it rewards:
- "latest wind energy outlook" matches every yearly edition equally well
  lexically; only a recency prior ranks the newest edition first.
- Some series were discontinued, so "latest" means the newest *matching* edition,
  not the newest document in the corpus — recency must be relative to the
  candidates, not global.
- Year-specific queries ("wind energy outlook 2019") must NOT be pushed toward
  recent editions, so a blanket recency boost hurts: the boost has to be gated
  on temporal intent in the query.
"""
from __future__ import annotations

import random

from search_rsi.problems.base import Query, SearchProblem, round_robin_splits
from search_rsi.types import Document

TOPICS = [
    ("wind energy", "turbine capacity, offshore leasing and grid connection"),
    ("solar energy", "panel efficiency, rooftop adoption and storage"),
    ("freight shipping", "container rates, port congestion and fuel costs"),
    ("housing market", "mortgage rates, construction starts and rents"),
    ("semiconductor supply", "fab capacity, wafer prices and lead times"),
    ("coffee harvest", "arabica yields, rainfall and export volumes"),
    ("airline travel", "passenger demand, fleet orders and fares"),
    ("water utilities", "reservoir levels, pipe replacement and tariffs"),
    ("battery materials", "lithium prices, cathode chemistry and recycling"),
    ("urban transit", "ridership, fare policy and fleet electrification"),
    ("wheat markets", "planting area, export quotas and futures"),
    ("cloud computing", "data center buildout, pricing and power demand"),
]
PUBLISHERS = ["Harbor Institute", "Northline Research", "Civic Data Office"]
LATEST_TEMPLATES = ["latest {t} outlook", "most recent {t} report", "current {t} outlook", "newest {t} forecast"]


def build(seed: int = 19) -> SearchProblem:
    rng = random.Random(seed)
    docs: list[Document] = []
    series: dict[str, list[Document]] = {}
    for topic, themes in TOPICS:
        start = rng.randint(2012, 2016)
        end = rng.choice([2019, 2021, 2023, 2024, 2024])
        publisher = rng.choice(PUBLISHERS)
        for year in range(start, end + 1):
            figure = rng.randint(3, 97)
            doc = Document(
                doc_id=f"rep_{topic.replace(' ', '_')}_{year}",
                title=f"{topic.title()} Outlook {year}",
                text=(f"The {year} edition of the {publisher} {topic} outlook reviews {themes}. "
                      f"Headline figure for the year: {figure} percent."),
                metadata={"topic": topic, "date": year},
            )
            docs.append(doc)
            series.setdefault(topic, []).append(doc)

    queries: list[Query] = []
    for topic, _ in TOPICS:
        editions = series[topic]
        for j in range(4):
            text = LATEST_TEMPLATES[j].format(t=topic)
            qrels = {editions[-1].doc_id: 2, editions[-2].doc_id: 1}
            queries.append(Query(qid=f"nq{len(queries):03d}", text=text, qrels=qrels, intent="latest"))
        for doc in rng.sample(editions[:-1], 3):
            year = doc.metadata["date"]
            queries.append(Query(qid=f"nq{len(queries):03d}", text=f"{topic} outlook {year}",
                                 qrels={doc.doc_id: 2}, intent="dated"))

    problem = SearchProblem(
        name="news_freshness",
        use_case="time-sensitive search where recency intent must be detected",
        description=__doc__.strip().splitlines()[0],
        documents=docs,
        queries=queries,
        splits=round_robin_splits(queries, group=lambda q: q.intent),
        primary_metric="ndcg@10",
    )
    problem.validate()
    return problem
