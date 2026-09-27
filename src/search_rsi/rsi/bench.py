"""The benchmark that has to exist before anything is claimed (plan.md §7), run on
every search use case at once, with a control arm for each RSI level:

1. Inner loop (rewrite memory): the baseline pipeline with vs. without rules
   learned on train, scored on held-out eval.
2. Outer loop (pipeline optimizer): the evolved champion vs. a random-search
   champion with the same evaluation budget vs. the baseline, all chosen on
   meta_val and scored once on held-out eval — plus the champion with its
   memory switched off, so the memory's share of the win is visible.
3. Transfer (cross-problem memory): leave-one-out — each problem takes the
   champions of its 3 most similar *other* problems, keeps the one that scores
   best on its own train split (3 evaluations instead of a search), and ships
   it; scored on held-out eval.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from search_rsi.pipeline import BASELINE_PIPELINE
from search_rsi.problems import load_problem
from search_rsi.rsi.evaluate import evaluate_pipeline
from search_rsi.rsi.learn import learn_rewrites
from search_rsi.rsi.optimize import optimize_pipeline
from search_rsi.rsi.transfer import ConfigMemory


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0


def run_benchmark(problem_names: list[str], iterations: int = 40, seeds: tuple[int, ...] = (0, 1, 2),
                  memory_dir: Path | None = None) -> dict:
    """Seeds only affect the optimizer arms (the rest is deterministic); evolved
    vs. random is reported as a mean over seeds, since one seed of a 40-step
    search is an anecdote, not a comparison."""
    results: dict = {"iterations": iterations, "seeds": list(seeds), "problems": {}}
    champions = {}
    for name in problem_names:
        started = time.perf_counter()
        problem = load_problem(name)
        held = "held_out_eval"

        # Level 1: inner loop, baseline pipeline, memory on vs. off.
        trace = learn_rewrites(problem, BASELINE_PIPELINE)
        base_eval = evaluate_pipeline(problem, BASELINE_PIPELINE, held)
        base_mem_eval = evaluate_pipeline(problem, replace(BASELINE_PIPELINE, use_learned_rewrites=True),
                                          held, trace.rules)
        # plan.md §7: the same training sequence with memory off is the control arm
        no_memory = evaluate_pipeline(problem, BASELINE_PIPELINE, "train").per_query
        no_memory_curve = [no_memory[q.qid] for q in problem.split("train")]
        half = len(trace.online_scores) // 2

        # Level 2: outer loop vs. random search, same budget, chosen on meta_val.
        runs = []
        for seed in seeds:
            evolved = optimize_pipeline(problem, iterations, seed, strategy="evolve")
            random_arm = optimize_pipeline(problem, iterations, seed, strategy="random")
            evolved_eval = evaluate_pipeline(problem, evolved.champion_config, held, evolved.champion_rules)
            random_eval = evaluate_pipeline(problem, random_arm.champion_config, held, random_arm.champion_rules)
            no_mem = evaluate_pipeline(problem, replace(evolved.champion_config, use_learned_rewrites=False), held)
            runs.append({
                "seed": seed,
                "evolved_champion": evolved.champion_config.label(),
                "evolved_config": evolved.champion_config.to_dict(),
                "evolved_meta_val": evolved.champion.score.mean_score,
                "evolved_accepted": f"{evolved.n_accepted}/{evolved.n_proposed}",
                "random_champion": random_arm.champion_config.label(),
                "random_meta_val": random_arm.champion.score.mean_score,
                "held_out_evolved": evolved_eval.mean_score,
                "held_out_evolved_all_metrics": evolved_eval.metrics,
                "held_out_evolved_memory_off": no_mem.mean_score,
                "held_out_random": random_eval.mean_score,
                "latency_ms_evolved": evolved_eval.mean_latency_ms,
                "pareto_front": [
                    {"config": e.config.label(), "meta_val": e.score.mean_score,
                     "latency_ms": e.score.mean_latency_ms}
                    for e in evolved.archive.pareto_front()
                ],
            })
            if seed == seeds[0]:
                champions[name] = (problem, evolved, evolved_eval)

        def agg(key: str) -> dict:
            xs = [r[key] for r in runs]
            return {"mean": _mean(xs), "std": _std(xs)}

        results["problems"][name] = {
            "summary": problem.summary(),
            "primary_metric": problem.primary_metric,
            "baseline": base_eval.mean_score,
            "baseline_all_metrics": base_eval.metrics,
            "latency_ms_baseline": base_eval.mean_latency_ms,
            "memory": {
                "n_rules": len(trace.rules),
                "n_demoted": len(trace.demoted),
                "n_rejected": trace.rejected,
                "rules": trace.rules.to_json(),
                "held_out_with_memory": base_mem_eval.mean_score,
                "train_online_curve": trace.online_scores,
                "train_no_memory_curve": no_memory_curve,
                "train_second_half_memory": _mean(trace.online_scores[half:]),
                "train_second_half_no_memory": _mean(no_memory_curve[half:]),
            },
            "optimizer": {
                "held_out_evolved": agg("held_out_evolved"),
                "held_out_evolved_memory_off": agg("held_out_evolved_memory_off"),
                "held_out_random": agg("held_out_random"),
                "runs": runs,
            },
            "seconds": time.perf_counter() - started,
        }

    # Level 3: leave-one-out transfer through the cross-problem config memory.
    if len(champions) > 1:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            memory = ConfigMemory(Path(memory_dir or tmp) / "champions.json")
            for name, (problem, evolved, evolved_eval) in champions.items():
                memory.record(problem, evolved.champion_config, evolved.champion.score.mean_score,
                              evolved_eval.mean_score)
            for name, (problem, _, _) in champions.items():
                probes = []
                for source, config, dist in memory.suggest(problem, k=3):
                    rules = learn_rewrites(problem, config).rules if config.use_learned_rewrites else None
                    probes.append((evaluate_pipeline(problem, config, "train", rules).mean_score,
                                   -dist, source, config, dist, rules))
                _, _, source, config, dist, rules = max(probes, key=lambda p: (p[0], p[1]))
                score = evaluate_pipeline(problem, config, "held_out_eval", rules).mean_score
                results["problems"][name]["transfer"] = {
                    "from": source, "config": config.label(), "distance": dist, "held_out": score,
                    "candidates": [p[2] for p in probes],
                }
    return results


def render_markdown(results: dict) -> str:
    seeds = results["seeds"]
    lines = [
        "# search-rsi benchmark",
        "",
        f"Optimizer budget: {results['iterations']} evaluations per arm; optimizer columns are the "
        f"mean ± std over seeds {seeds}. Every score is on **held-out eval** unless marked meta_val: "
        "held-out queries never influenced a rule, an accept/reject decision, or a champion.",
        "",
        "## Headline",
        "",
        "| problem | metric | baseline | + memory | evolved | evolved, memory off | random search | transfer (3 probes) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def ms(d: dict) -> str:
        return f"{d['mean']:.3f} ± {d['std']:.3f}"

    for name, r in results["problems"].items():
        opt, mem = r["optimizer"], r["memory"]
        transfer = r.get("transfer", {}).get("held_out")
        lines.append(
            f"| {name} | {r['primary_metric']} | {r['baseline']:.3f} | {mem['held_out_with_memory']:.3f} | "
            f"**{ms(opt['held_out_evolved'])}** | {ms(opt['held_out_evolved_memory_off'])} | "
            f"{ms(opt['held_out_random'])} | {'' if transfer is None else f'{transfer:.3f}'} |"
        )
    lines += ["", f"## Champions (chosen on meta_val, seed {seeds[0]})", ""]
    for name, r in results["problems"].items():
        run = r["optimizer"]["runs"][0]
        lines.append(f"- **{name}**: `{run['evolved_champion']}` (meta_val {run['evolved_meta_val']:.3f}, "
                     f"accepted {run['evolved_accepted']}; random arm `{run['random_champion']}` "
                     f"meta_val {run['random_meta_val']:.3f}); latency {r['latency_ms_baseline']:.2f} → "
                     f"{run['latency_ms_evolved']:.2f} ms/query")
        if "transfer" in r:
            t = r["transfer"]
            lines.append(f"  - transfer: shipped `{t['config']}` from {t['from']} (profile distance "
                         f"{t['distance']:.2f}; probed {', '.join(t['candidates'])})")
    lines += ["", "## Learned rewrite memory (baseline pipeline, train split)", ""]
    for name, r in results["problems"].items():
        mem = r["memory"]
        shown = ", ".join(f"`{x['source']}→{'/'.join(x['targets'])}`" for x in mem["rules"][:8])
        lines.append(f"- **{name}**: {mem['n_rules']} rules ({mem['n_demoted']} demoted on probation, "
                     f"{mem['n_rejected']} rejected at promotion); second half of the train "
                     f"stream: {mem['train_second_half_memory']:.3f} with memory vs. "
                     f"{mem['train_second_half_no_memory']:.3f} without. {shown}")
    return "\n".join(lines) + "\n"


def write_report(results: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "benchmark.md").write_text(render_markdown(results))
    return out_dir
