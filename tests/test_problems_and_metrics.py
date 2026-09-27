import json

import pytest

from search_rsi.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k, success_at_k
from search_rsi.problems import BUILTIN_PROBLEMS, SPLIT_NAMES, load_beir_dir, load_problem
from search_rsi.problems import product


def test_metrics_on_hand_computed_rankings():
    qrels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], qrels) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], qrels) < ndcg_at_k(["a", "c", "b"], qrels) < 1.0
    assert mrr_at_k(["c", "b"], qrels) == pytest.approx(0.5)
    assert recall_at_k(["a"], qrels) == pytest.approx(0.5)
    assert success_at_k(["b", "a"], qrels, 1) == 0.0  # b is relevant but not the best
    assert success_at_k(["a"], qrels, 1) == 1.0
    assert ndcg_at_k(["a"], {}) == 0.0


@pytest.mark.parametrize("name", list(BUILTIN_PROBLEMS))
def test_every_builtin_problem_is_valid_and_split_three_ways(name):
    problem = load_problem(name)
    problem.validate()
    for split in SPLIT_NAMES:
        assert len(problem.split(split)) >= 8, f"{name}/{split} too small to gate decisions on"
    all_ids = [qid for split in SPLIT_NAMES for qid in problem.splits[split]]
    assert len(all_ids) == len(set(all_ids))


def test_generators_are_deterministic():
    a, b = product.build(), product.build()
    assert [d.text for d in a.documents] == [d.text for d in b.documents]
    assert [(q.text, q.qrels) for q in a.queries] == [(q.text, q.qrels) for q in b.queries]
    assert a.splits == b.splits


def test_user_vocabulary_is_absent_from_the_catalog():
    """The synonym gap is real by construction: no catalog text says 'sneakers'."""
    problem = load_problem("product_search")
    assert any("sneakers" in q.text for q in problem.queries)
    assert not any("sneakers" in f"{d.title} {d.text}".lower() for d in problem.documents)


def test_beir_directory_loader(tmp_path):
    (tmp_path / "qrels").mkdir()
    docs = [{"_id": f"d{i}", "title": f"title {i}", "text": f"document number {i} about topic{i % 3}"}
            for i in range(30)]
    queries = [{"_id": f"q{i}", "text": f"topic{i % 3} number {i}"} for i in range(30)]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    rows = ["query-id\tcorpus-id\tscore"] + [f"q{i}\td{i}\t1" for i in range(30)] + ["q0\tmissing\t1"]
    (tmp_path / "qrels" / "test.tsv").write_text("\n".join(rows))

    problem = load_beir_dir(tmp_path, name="toy")
    assert len(problem.documents) == 30 and len(problem.queries) == 30
    assert problem.documents[0].title == "title 0"
    assert sum(len(v) for v in problem.splits.values()) == 30
    assert load_beir_dir(tmp_path, name="toy").splits == problem.splits  # hash split is stable
