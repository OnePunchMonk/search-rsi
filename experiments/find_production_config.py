"""The actual product loop: given a search problem (a corpus + a benchmarked task
set), search over retrieval algorithms/hyperparameters/generative techniques, pick
the option that's best-scoring on held-out eval among those at or near the lowest
latency, and export it as something a production server can load.

Usage: python experiments/find_production_config.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from search_rsi.benchmarks_corpus import load_task_splits
from search_rsi.meta import evaluate_variant, export_variant, run_meta_loop
from search_rsi.meta.archive import ArchiveEntry

ITERATIONS = 40
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "deploy" / "search_rsi_production"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    splits = load_task_splits()
    result = run_meta_loop(splits["meta_val"], iterations=ITERATIONS, seed=0)

    # Re-score every Pareto-front candidate on held-out eval — the champion on
    # meta_val is not automatically the champion for deployment; only held-out
    # performance is allowed to make that call (docs/META_RSI.md Guardrail 2).
    candidates = result.archive.pareto_front()
    held_out_results = [
        (entry, evaluate_variant(entry.config, splits["held_out_eval"]))
        for entry in candidates
    ]

    print("Pareto-front candidates re-scored on held-out eval:")
    for entry, held_out_score in held_out_results:
        print(f"  {entry.config.label():<45} held_out_score={held_out_score.mean_score:.3f}  "
              f"latency={held_out_score.mean_latency_ms:.3f}ms")

    # Best score wins; ties broken by lower latency, matching Archive.champion.
    best_entry, best_held_out = max(
        held_out_results, key=lambda pair: (pair[1].mean_score, -pair[1].mean_latency_ms)
    )
    export_entry = ArchiveEntry(best_entry.config, best_held_out)
    out_path = export_variant(export_entry, args.out_dir)

    print(f"\nExported production config to {out_path}/")
    print(f"  config:    {best_entry.config.label()}")
    print(f"  score:     {best_held_out.mean_score:.3f} (on held-out eval, {best_held_out.n_tasks} tasks)")
    print(f"  latency:   {best_held_out.mean_latency_ms:.3f}ms/query")
    print(f"  tool calls: {best_held_out.mean_tool_calls:.2f}/query")


if __name__ == "__main__":
    main()
