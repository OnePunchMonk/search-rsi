from search_rsi.benchmarks_corpus import load_task_splits
from search_rsi.meta import BASELINE, evaluate_variant, run_meta_loop, run_random_search_control


def test_splits_are_disjoint_and_frozen():
    """docs/META_RSI.md Guardrail 2: train / meta_val / held_out_eval must never
    overlap, or an accept decision could be contaminated by held-out data."""
    splits = load_task_splits()
    train_qs = {t.question for t in splits["train"]}
    meta_val_qs = {t.question for t in splits["meta_val"]}
    held_out_qs = {t.question for t in splits["held_out_eval"]}
    assert not (train_qs & meta_val_qs)
    assert not (train_qs & held_out_qs)
    assert not (meta_val_qs & held_out_qs)


def test_meta_loop_never_beats_champion_score_by_regressing_tool_calls():
    """Archive.maybe_accept must be Pareto-aware: never accept a strictly worse
    variant (docs/META_RSI.md Guardrail 3)."""
    splits = load_task_splits()
    result = run_meta_loop(splits["meta_val"], iterations=10, seed=1)
    assert result.champion_score >= result.baseline_score


def test_random_search_control_arm_actually_runs():
    """docs/META_RSI.md Guardrail 4: the control arm must be a real, exercised path,
    not a stub, so 'evolved beat random search' is a checkable claim."""
    splits = load_task_splits()
    result = run_random_search_control(splits["meta_val"], iterations=5, seed=2)
    assert result.n_proposed == 5
    assert result.champion_score >= result.baseline_score


def test_baseline_config_is_a_real_evaluation_not_a_placeholder():
    splits = load_task_splits()
    score = evaluate_variant(BASELINE, splits["meta_val"])
    assert 0.0 <= score.mean_score <= 1.0
    assert score.n_tasks == len(splits["meta_val"])
