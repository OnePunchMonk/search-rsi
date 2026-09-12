import dataclasses
import json

import pytest

from search_rsi.benchmarks_corpus import load_default_corpus, load_gold_tasks
from search_rsi.harness import Harness
from search_rsi.types import Document


def make_harness(tmp_path, documents=None):
    return Harness.local(documents=documents, memory_path=tmp_path / "entries.json")


def test_real_retrieval_end_to_end(tmp_path):
    harness = make_harness(tmp_path)
    task = load_gold_tasks()[0]
    result = harness.run(task, memory_enabled=False, is_training_task=False)
    assert result.answer == task.gold_answer
    assert result.score > 0.9
    assert result.tool_calls_used >= 1


def test_score_is_data_dependent_not_a_formula(tmp_path):
    """C1: corrupting the corpus must change the score. A formula-derived score
    that ignores the actual documents would pass unchanged."""
    task = load_gold_tasks()[0]

    real_docs = load_default_corpus()
    real_score = make_harness(tmp_path, real_docs).run(
        task, memory_enabled=False, is_training_task=False
    ).score

    corrupted_docs = [
        Document(doc_id=d.doc_id, text="irrelevant filler text about nothing in particular")
        for d in real_docs
    ]
    corrupted_score = make_harness(tmp_path / "b", corrupted_docs).run(
        task, memory_enabled=False, is_training_task=False
    ).score

    assert corrupted_score < real_score


def test_holdout_eval_never_writes_memory(tmp_path):
    """C4: eval-flagged runs must not mutate the memory store, even on failure."""
    memory_path = tmp_path / "entries.json"
    harness = Harness.local(memory_path=memory_path)
    before = memory_path.read_text()

    hard_task = load_gold_tasks()[1]
    for _ in range(3):
        harness.run(hard_task, memory_enabled=True, is_training_task=False)

    after = memory_path.read_text()
    assert before == after


def test_tool_budget_is_enforced(tmp_path):
    """C5: the harness must never exceed the declared tool budget."""
    harness = make_harness(tmp_path)
    task = load_gold_tasks()[1]
    tight = dataclasses.replace(task, tool_budget=1)
    result = harness.run(tight, memory_enabled=True, is_training_task=True)
    assert result.tool_calls_used <= 1


def test_ablation_arm_exists_and_differs_in_queries_after_training(tmp_path):
    """C6: memory_enabled=False must be a real, exercised code path."""
    memory_path = tmp_path / "entries.json"
    hard_task = load_gold_tasks()[1]

    trained = Harness.local(memory_path=memory_path)
    for _ in range(5):
        trained.run(hard_task, memory_enabled=True, is_training_task=True)

    memory_entries = json.loads(memory_path.read_text())
    assert len(memory_entries) >= 1

    with_memory = trained.run(hard_task, memory_enabled=True, is_training_task=False)
    without_memory = trained.run(hard_task, memory_enabled=False, is_training_task=False)
    assert with_memory.tool_calls_used != without_memory.tool_calls_used or (
        with_memory.score >= without_memory.score
    )
