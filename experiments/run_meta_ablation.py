"""RSI-for-RSI benchmark: does the meta-loop find a better harness config (any
retrieval algorithm, any hyperparameters, generative query expansion on or off) than
a random-search control with the same evaluation budget, and does the winner
generalize to held-out eval tasks it never touched? (docs/META_RSI.md §Benchmark)

Usage: python experiments/run_meta_ablation.py
"""
from __future__ import annotations

from search_rsi.benchmarks_corpus import load_task_splits
from search_rsi.meta import BASELINE, evaluate_variant, run_meta_loop, run_random_search_control

ITERATIONS = 30


def report(name: str, score) -> None:
    print(f"{name:<14} score={score.mean_score:.3f}  calls={score.mean_tool_calls:.2f}  "
          f"latency={score.mean_latency_ms:.3f}ms")


def main() -> None:
    splits = load_task_splits()
    meta_val = splits["meta_val"]
    held_out = splits["held_out_eval"]

    evolved = run_meta_loop(meta_val, iterations=ITERATIONS, seed=0)
    random_control = run_random_search_control(meta_val, iterations=ITERATIONS, seed=0)

    print("== meta-val (used for accept/reject during search) ==")
    print(f"baseline config:      {BASELINE.label()}")
    print(f"evolved champion:     {evolved.champion_config.label()} "
          f"({evolved.n_accepted}/{evolved.n_proposed} accepted)")
    print(f"random-search champ:  {random_control.champion_config.label()} "
          f"({random_control.n_accepted}/{random_control.n_proposed} accepted)")

    print("\nevolved-arm Pareto front (score, latency tradeoffs actually available):")
    for e in evolved.archive.pareto_front():
        print(f"  {e.config.label():<45} score={e.score.mean_score:.3f}  "
              f"latency={e.score.mean_latency_ms:.3f}ms")

    print("\n== held-out eval (touched once, here, never used for accept/reject) ==")
    baseline_eval = evaluate_variant(BASELINE, held_out)
    evolved_eval = evaluate_variant(evolved.champion_config, held_out)
    random_eval = evaluate_variant(random_control.champion_config, held_out)
    report("baseline", baseline_eval)
    report("evolved", evolved_eval)
    report("random-search", random_eval)

    if evolved_eval.mean_score <= baseline_eval.mean_score:
        print("\nNo generalized quality improvement over baseline on held-out eval yet.")
    elif evolved_eval.mean_score <= random_eval.mean_score:
        print("\nEvolved arm did not beat random search — the archive/mutation search isn't earning "
              "its keep over this search space yet.")
    else:
        print("\nEvolved arm beat both baseline and random search on held-out eval.")


if __name__ == "__main__":
    main()
