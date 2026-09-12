"""Runs the plan.md §7 benchmark: same task sequence, memory on vs. off.

Usage: python experiments/run_ablation.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from search_rsi.benchmarks_corpus import load_gold_tasks
from search_rsi.harness import Harness


def run_sequence(memory_enabled: bool) -> list[tuple[float, int]]:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness.local(memory_path=Path(tmp) / "entries.json")
        results = []
        for task in load_gold_tasks() * 3:  # repeat the small gold set to form a sequence
            r = harness.run(task, memory_enabled=memory_enabled, is_training_task=True)
            results.append((r.score, r.tool_calls_used))
        return results


def main() -> None:
    with_memory = run_sequence(memory_enabled=True)
    without_memory = run_sequence(memory_enabled=False)

    print(f"{'task#':>6} {'score(mem)':>11} {'calls(mem)':>11} {'score(no-mem)':>14} {'calls(no-mem)':>14}")
    for i, ((s1, c1), (s2, c2)) in enumerate(zip(with_memory, without_memory)):
        print(f"{i:>6} {s1:>11.2f} {c1:>11} {s2:>14.2f} {c2:>14}")


if __name__ == "__main__":
    main()
