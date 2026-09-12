import json

from search_rsi.benchmarks_corpus import load_task_splits
from search_rsi.harness import Harness
from search_rsi.meta import BASELINE, evaluate_variant, export_variant
from search_rsi.meta.archive import ArchiveEntry


def test_export_produces_a_loadable_config(tmp_path):
    splits = load_task_splits()
    score = evaluate_variant(BASELINE, splits["held_out_eval"])
    out_dir = export_variant(ArchiveEntry(BASELINE, score), tmp_path / "prod")

    config = json.loads((out_dir / "config.json").read_text())
    assert config["retriever"] == "bm25"

    harness = Harness.local(memory_path=tmp_path / "mem.json", **config)
    task = splits["held_out_eval"][0]
    result = harness.run(task, memory_enabled=False, is_training_task=False)
    assert result.score >= 0.0  # real run, not a crash


def test_pareto_front_excludes_dominated_entries():
    from search_rsi.meta.archive import Archive
    from search_rsi.meta.evaluate import VariantScore

    archive = Archive(ArchiveEntry(BASELINE, VariantScore(mean_score=0.5, mean_tool_calls=2, mean_latency_ms=10, n_tasks=4)))
    dominated = ArchiveEntry(BASELINE, VariantScore(mean_score=0.4, mean_tool_calls=2, mean_latency_ms=20, n_tasks=4))
    better_but_slower = ArchiveEntry(BASELINE, VariantScore(mean_score=0.6, mean_tool_calls=2, mean_latency_ms=30, n_tasks=4))
    archive.entries.append(dominated)
    archive.entries.append(better_but_slower)

    front_scores = {e.score.mean_score for e in archive.pareto_front()}
    assert 0.4 not in front_scores  # dominated on both score and latency
    assert 0.5 in front_scores  # fastest
    assert 0.6 in front_scores  # highest score
