"""Multi-hop term provenance: the original search-rsi benchmark, as a ranking problem.

Each question asks where a term originated or how a successor body redefined it;
the answering document is graded 2 and the other document of the reasoning chain
1. Decoy documents share surface vocabulary with the answer (see
benchmarks/corpus/generate_corpus.py), and the frozen split is the one the agentic
Harness's meta-loop already uses.
"""
from __future__ import annotations

import json

from search_rsi.benchmarks_corpus import CORPUS_DIR, load_default_corpus, load_gold_tasks
from search_rsi.problems.base import Query, SearchProblem


def build() -> SearchProblem:
    tasks = load_gold_tasks()
    queries = [
        Query(
            qid=f"mh{i:03d}",
            text=t.question,
            qrels={d: (2 if d == t.gold_answer else 1) for d in t.gold_doc_ids},
            intent=t.task_type,
        )
        for i, t in enumerate(tasks)
    ]
    index_splits = json.loads((CORPUS_DIR / "splits.json").read_text())
    problem = SearchProblem(
        name="multihop_provenance",
        use_case="multi-hop provenance questions with decoy documents",
        description=__doc__.strip().splitlines()[0],
        documents=load_default_corpus(),
        queries=queries,
        splits={name: [queries[i].qid for i in idxs] for name, idxs in index_splits.items()},
        primary_metric="ndcg@10",
    )
    problem.validate()
    return problem
