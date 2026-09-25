import json
from dataclasses import replace

import pytest

from search_rsi.cli import main as cli_main
from search_rsi.memory import RewriteMemory
from search_rsi.pipeline import BASELINE_PIPELINE, PipelineConfig
from search_rsi.problems import load_problem
from search_rsi.rsi import (
    ConfigMemory,
    evaluate_pipeline,
    export_pipeline,
    learn_rewrites,
    optimize_pipeline,
    profile_problem,
)


def test_score_is_data_dependent():
    """C1 for pipelines: corrupting the judgments must change the score."""
    problem = load_problem("faq_support")
    real = evaluate_pipeline(problem, BASELINE_PIPELINE, "meta_val").mean_score
    shuffled = [replace(q, qrels={"faq_pause_subscription_personal": 2}) for q in problem.split("meta_val")]
    assert evaluate_pipeline(problem, BASELINE_PIPELINE, shuffled).mean_score != real


def test_learned_rewrites_generalize_to_held_out():
    """The inner-loop claim: rules learned on train lift held-out queries they never saw."""
    problem = load_problem("product_search")
    trace = learn_rewrites(problem, BASELINE_PIPELINE)
    assert any(r.source == "sneakers" for r in trace.rules)
    without = evaluate_pipeline(problem, BASELINE_PIPELINE, "held_out_eval").mean_score
    with_memory = evaluate_pipeline(problem, replace(BASELINE_PIPELINE, use_learned_rewrites=True),
                                    "held_out_eval", trace.rules).mean_score
    assert with_memory > without + 0.05


def test_rewrite_learning_refuses_non_train_splits():
    with pytest.raises(ValueError):
        learn_rewrites(load_problem("faq_support"), BASELINE_PIPELINE, split="held_out_eval")


def test_eval_never_writes_memory(tmp_path):
    """C4: evaluating any split, with rules, leaves the persisted memory untouched."""
    problem = load_problem("faq_support")
    memory = RewriteMemory(tmp_path / "rewrites.json")
    memory.write(problem.name, learn_rewrites(problem, BASELINE_PIPELINE).rules)
    before = memory.path.read_bytes()
    for split in ("train", "meta_val", "held_out_eval"):
        evaluate_pipeline(problem, replace(BASELINE_PIPELINE, use_learned_rewrites=True), split,
                          memory.read(problem.name))
    assert memory.path.read_bytes() == before


def test_optimizer_only_ever_sees_meta_val(monkeypatch):
    """Guardrail 2: accept/reject is decided on meta_val; held-out is never touched,
    and train is only read by the rewrite learner."""
    problem = load_problem("faq_support")
    seen = []
    original = type(problem).split
    monkeypatch.setattr(type(problem), "split", lambda self, name: seen.append(name) or original(self, name))
    optimize_pipeline(problem, iterations=6, seed=0)
    assert "held_out_eval" not in seen
    assert set(seen) <= {"meta_val", "train"}


def test_optimizer_never_crowns_a_worse_champion_and_random_arm_runs():
    problem = load_problem("news_freshness")
    for strategy in ("evolve", "random"):
        result = optimize_pipeline(problem, iterations=8, seed=3, strategy=strategy)
        assert result.n_proposed == 8
        assert result.champion.score.mean_score >= result.baseline.mean_score
    evolved = optimize_pipeline(problem, iterations=25, seed=0)
    assert evolved.champion.score.mean_score > evolved.baseline.mean_score  # recency is findable


def test_mutation_surface_excludes_the_grader():
    fields = set(PipelineConfig.__dataclass_fields__)
    assert not fields & {"qrels", "primary_metric", "splits", "grader", "metric"}


def test_transfer_memory_suggests_the_most_similar_problem(tmp_path):
    memory = ConfigMemory(tmp_path / "champions.json")
    news, entity, product = (load_problem(n) for n in ("news_freshness", "entity_lookup", "product_search"))
    memory.record(entity, PipelineConfig(retriever="charngram"), 1.0, 1.0)
    memory.record(product, PipelineConfig(filter_mode="soft"), 0.9, 0.9)
    assert profile_problem(entity).query_typo_rate > profile_problem(product).query_typo_rate
    assert memory.suggest(entity) == [("product_search", PipelineConfig(filter_mode="soft"),
                                       pytest.approx(memory.suggest(entity)[0][2]))]
    assert len(memory.suggest(news, k=5)) == 2


def test_export_round_trip(tmp_path):
    problem = load_problem("code_search")
    config = PipelineConfig(analyzer="code", use_learned_rewrites=True)
    rules = learn_rewrites(problem, config).rules
    score = evaluate_pipeline(problem, config, "held_out_eval", rules)
    out = export_pipeline(tmp_path / "prod", problem.name, config, rules, score)

    import importlib.util

    spec = importlib.util.spec_from_file_location("serve", out / "serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    pipeline = serve.load_pipeline(problem.documents)
    query = problem.split("held_out_eval")[0]
    assert [d for d, _ in serve.search(pipeline, query.text, 100)] == \
        [d for d, _ in pipeline.search(query.text, 100).hits]
    assert json.loads((out / "manifest.json").read_text())["held_out"]["primary"] == pytest.approx(score.mean_score)


def test_cli_smoke(tmp_path, capsys):
    cli_main(["--memory-dir", str(tmp_path), "problems"])
    cli_main(["--memory-dir", str(tmp_path), "learn", "faq_support"])
    cli_main(["--memory-dir", str(tmp_path), "search", "faq_support", "terminate my membership",
              "--config", '{"use_learned_rewrites": true}', "-k", "1"])
    out = capsys.readouterr().out
    assert "terminate -> cancel" in " ".join(out.split())  # learned, persisted...
    assert "executed query: 'terminate my membership cancel'" in out  # ...and applied
    assert "faq_cancel_" in out.splitlines()[-1]
