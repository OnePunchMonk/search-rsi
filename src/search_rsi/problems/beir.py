"""Bring your own search problem: load any BEIR-format dataset directory.

BEIR (Thakur et al., 2021) is the de-facto interchange format for retrieval
benchmarks — SciFact, NFCorpus, FiQA, ArguAna, TREC-COVID and more ship in it:

    corpus.jsonl    {"_id": ..., "title": ..., "text": ...}  one per line
    queries.jsonl   {"_id": ..., "text": ...}
    qrels/test.tsv  query-id <tab> corpus-id <tab> score   (header line)

The directory's qrels are the grader (plan.md C1); queries are split into
train / meta_val / held_out_eval by a content hash of their ids, so the split is
frozen and reproducible without writing anything back into the dataset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from search_rsi.problems.base import Query, SearchProblem, hash_splits
from search_rsi.types import Document


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_beir_dir(path: str | Path, qrels_file: str | None = None, name: str | None = None) -> SearchProblem:
    root = Path(path)
    docs = [
        Document(doc_id=str(d["_id"]), title=d.get("title", "") or "", text=d.get("text", "") or "",
                 metadata=d.get("metadata") or {})
        for d in _read_jsonl(root / "corpus.jsonl")
    ]
    query_text = {str(q["_id"]): q["text"] for q in _read_jsonl(root / "queries.jsonl")}

    qrels_path = root / "qrels" / (qrels_file or "test.tsv")
    if not qrels_path.exists():
        candidates = sorted((root / "qrels").glob("*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"no qrels/*.tsv under {root}")
        qrels_path = candidates[0]
    doc_ids = {d.doc_id for d in docs}
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open() as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3 or not row[2].lstrip("-").isdigit():
                continue  # header or malformed line
            qid, doc_id, grade = row[0], row[1], int(row[2])
            if doc_id in doc_ids and qid in query_text:
                qrels.setdefault(qid, {})[doc_id] = grade

    queries = [Query(qid=qid, text=query_text[qid], qrels=rels) for qid, rels in sorted(qrels.items())
               if any(g > 0 for g in rels.values())]
    problem = SearchProblem(
        name=name or root.name,
        use_case="imported BEIR-format dataset",
        description=f"BEIR-format dataset loaded from {root}",
        documents=docs,
        queries=queries,
        splits=hash_splits(q.qid for q in queries),
        primary_metric="ndcg@10",
    )
    problem.validate()
    return problem
