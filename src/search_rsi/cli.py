"""search-rsi command line.

    search-rsi problems                         list the search use cases
    search-rsi search product_search "red sneakers under $80" [--config cfg.json]
    search-rsi eval code_search --config cfg.json --split meta_val
    search-rsi learn faq_support                learn + persist rewrite rules (train split)
    search-rsi optimize product_search --export deploy/product
    search-rsi bench --out reports/

PROBLEM is a built-in name or a path to a BEIR-format directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from search_rsi.memory import RewriteMemory
from search_rsi.pipeline import BASELINE_PIPELINE, PipelineConfig, SearchPipeline
from search_rsi.problems import list_problems, load_problem

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory" / "store"


def _config(arg: str | None) -> PipelineConfig:
    if not arg:
        return BASELINE_PIPELINE
    text = Path(arg).read_text() if Path(arg).is_file() else arg
    return PipelineConfig.from_dict(json.loads(text))


def cmd_problems(args) -> None:
    for name in list_problems():
        s = load_problem(name).summary()
        print(f"{name:22s} {s['documents']:4d} docs {s['queries']:4d} queries  "
              f"{s['primary_metric']:10s} {s['use_case']}")


def cmd_search(args) -> None:
    problem = load_problem(args.problem)
    config = _config(args.config)
    rules = RewriteMemory(args.memory_dir / "rewrites.json").read(problem.name)
    pipeline = SearchPipeline(problem.documents, config, rules, cache=problem.index_cache)
    response = pipeline.search(args.query, args.k)
    by_id = {d.doc_id: d for d in problem.documents}
    print(f"config: {config.label()}\nexecuted query: {response.query!r}")
    if response.constraints:
        print(f"constraints: {response.constraints}")
    for doc_id, score in response.hits:
        doc = by_id[doc_id]
        print(f"{score:8.4f}  {doc_id:28s} {(doc.title or doc.text)[:70]}")


def cmd_eval(args) -> None:
    from search_rsi.rsi import evaluate_pipeline

    problem = load_problem(args.problem)
    config = _config(args.config)
    rules = RewriteMemory(args.memory_dir / "rewrites.json").read(problem.name)
    score = evaluate_pipeline(problem, config, args.split, rules if config.use_learned_rewrites else None)
    print(json.dumps({"config": config.label(), "split": args.split, "primary": problem.primary_metric,
                      **score.metrics, "latency_ms": score.mean_latency_ms}, indent=2))


def cmd_learn(args) -> None:
    from search_rsi.rsi import learn_rewrites

    problem = load_problem(args.problem)
    memory = RewriteMemory(args.memory_dir / "rewrites.json")
    trace = learn_rewrites(problem, _config(args.config), memory.read(problem.name))
    memory.write(problem.name, trace.rules)
    print(f"{len(trace.rules)} rules for {problem.name} ({len(trace.promoted)} promoted, "
          f"{len(trace.demoted)} demoted, {trace.rejected} rejected) -> {memory.path}")
    for rule in trace.rules:
        print(f"  {rule.source:>16s} -> {' '.join(rule.targets):30s} support={rule.support}")


def cmd_optimize(args) -> None:
    from search_rsi.rsi import ConfigMemory, evaluate_pipeline, export_pipeline, optimize_pipeline

    problem = load_problem(args.problem)
    rewrite_memory = RewriteMemory(args.memory_dir / "rewrites.json")
    config_memory = ConfigMemory(args.memory_dir / "champions.json")
    warm = [config for _, config, _ in config_memory.suggest(problem, k=2)] if not args.cold else []
    result = optimize_pipeline(problem, args.iterations, args.seed, strategy=args.strategy,
                               warm_starts=warm, seed_rules=rewrite_memory.read(problem.name))
    champ = result.champion
    rules = result.champion_rules
    held = evaluate_pipeline(problem, champ.config, "held_out_eval", rules)
    base = evaluate_pipeline(problem, BASELINE_PIPELINE, "held_out_eval")
    print(f"warm starts: {[c.label() for c in warm] or 'none'}")
    print(f"accepted {result.n_accepted}/{result.n_proposed}; Pareto front on meta_val:")
    for e in result.archive.pareto_front():
        print(f"  {e.score.mean_score:.3f}  {e.score.mean_latency_ms:6.2f}ms  {e.config.label()}")
    print(f"champion: {champ.config.label()}")
    print(f"held-out {problem.primary_metric}: baseline {base.mean_score:.3f} -> champion {held.mean_score:.3f}")
    config_memory.record(problem, champ.config, champ.score.mean_score, held.mean_score)
    if rules is not None:
        rewrite_memory.write(problem.name, rules)
    if args.export:
        out = export_pipeline(args.export, problem.name, champ.config, rules, held, champ.score)
        print(f"exported deployable pipeline to {out}/")


def cmd_bench(args) -> None:
    from search_rsi.rsi.bench import render_markdown, run_benchmark, write_report

    names = args.problems or list_problems()
    results = run_benchmark(names, args.iterations, tuple(args.seeds))
    print(render_markdown(results))
    if args.out:
        print(f"wrote {write_report(results, args.out)}/benchmark.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="search-rsi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("problems").set_defaults(fn=cmd_problems)

    p = sub.add_parser("search")
    p.add_argument("problem")
    p.add_argument("query")
    p.add_argument("--config")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("eval")
    p.add_argument("problem")
    p.add_argument("--config")
    p.add_argument("--split", default="meta_val", choices=["train", "meta_val", "held_out_eval"])
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("learn")
    p.add_argument("problem")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("optimize")
    p.add_argument("problem")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategy", default="evolve", choices=["evolve", "random"])
    p.add_argument("--cold", action="store_true", help="ignore champions of similar problems")
    p.add_argument("--export", type=Path)
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("bench")
    p.add_argument("--problems", nargs="*")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_bench)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
