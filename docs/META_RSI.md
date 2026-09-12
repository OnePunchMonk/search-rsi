# RSI for RSI: evolving the harness itself

`search-rsi`'s inner loop (see `plan.md`, `docs/ARCHITECTURE.md`) improves at *tasks*
by accumulating strategy memory over fixed harness code. This document is about a
layer above that: a **meta-loop** that improves the *harness itself* — its retrieval
parameters, its planner logic, its verifier thresholds — across meta-iterations. That
is a different and much riskier kind of self-improvement, and this design leans
heavily on three prior systems that got the guardrails right:

- **Darwin Gödel Machine (DGM)** — an agent that rewrites its own code, validated
  empirically against a benchmark rather than proven correct, keeping an *archive* of
  accepted variants instead of always mutating the current best (avoids getting stuck
  in local optima; a "worse" variant can be a better stepping stone).
- **AlphaEvolve / FunSearch** — LLM-proposed code mutations, evaluated by a fixed,
  untouchable external scorer, accepted only on measured improvement.
- **ADAS (Automated Design of Agentic Systems)** — meta-agent designs agent
  *architectures* in code, discovered designs are meta-validated on tasks distinct
  from where they were found, to catch meta-level overfitting.

## Guardrail 1: the mutable surface is a closed, explicit set

Everything the meta-loop can touch is enumerated in `HarnessConfig`
(`src/search_rsi/meta/config.py`). v1 is deliberately numeric-only: BM25's `k1`/`b`,
the verifier's `min_overlap`. There is no field for the grader, the gold task set, or
the tool-budget accounting — a system that can edit its own exam will learn to edit
its exam, not to search better. Milestone 2 (below) grows this surface to short code
diffs in `agents/planner.py`, but the same principle holds: the diff target is never
`eval/grading.py`, `benchmarks/corpus/gold_tasks.json`, or the split boundaries.

## Guardrail 2: three-way split, frozen once

`benchmarks_corpus.load_task_splits()` fixes:

- **train** — feeds the *inner*-loop strategy memory (existing behavior, unchanged).
- **meta_val** — what the meta-loop's accept/reject gate (`Archive.maybe_accept`)
  scores candidates against. Never used by the inner loop, never reported as a final
  result.
- **held_out_eval** — touched exactly once, after the meta-loop has already picked a
  champion, purely to report whether the win generalizes. If a config wins on
  meta_val but loses on held_out_eval, that's meta-overfitting, and the honest
  report is "no," not a rerun with a friendlier split.

This mirrors `plan.md` C4 (eval tasks never write inner-loop memory) one level up:
held-out eval tasks must never influence any accept/reject decision, inner or outer.

## Guardrail 3: an archive, not a single champion

`src/search_rsi/meta/archive.py` keeps every accepted variant, and the next
mutation's parent is sampled uniformly from the archive rather than always taken from
the current best (`Archive.sample_parent`). `run_meta_loop(..., use_archive=False)`
reproduces plain hill-climbing for comparison. Acceptance is Pareto-aware: a
candidate that ties on score but uses fewer tool calls is accepted too, since
"fewer tool calls at equal quality" is the actual north-star metric (`plan.md` §1),
not raw score alone.

## Guardrail 4 (the meta-level ablation): beat random search, not just the baseline

`experiments/run_meta_ablation.py` runs two arms with the *same* evaluation budget:
the archive-based evolutionary search, and `run_random_search_control` (independently
sampled configs, no mutation, no archive). If the evolved arm doesn't beat the random
arm, the search mechanism isn't earning its keep — the config space alone would have
handed you the same result for the same compute. This is the meta-level version of
`plan.md` C6.

## Current status (honest)

Running `experiments/run_meta_ablation.py` today shows **0/20 accepted on both
arms** — the baseline config already scores 1.0 on the current `meta_val` split (2
tasks), so there is no headroom to improve into. This is a real ceiling effect from
the tiny benchmark corpus, not a bug in the loop: BM25 finds the right document
easily at this corpus size regardless of `k1`/`b`, and `min_overlap` doesn't bind. It
is being reported here rather than papered over, per the project's own C1 spirit —
don't claim an effect the number doesn't show.

**What would make the meta-loop's win real:**
1. A larger, harder benchmark corpus (`plan.md` §6, milestone still open) with enough
   lexical ambiguity that `k1`/`b`/`min_overlap` actually trade off precision vs.
   recall — ceiling effects at 6 tiny documents-per-task hide any config's edge.
2. More meta-val tasks (2 is too few to trust any accept decision statistically) —
   target at least ~15-20 before trusting a single accept/reject comparison, and add
   a minimum-sample-size / significance check to `Archive.maybe_accept` rather than
   accepting on a single evaluation.
3. Only then does growing the mutation surface to code diffs (Milestone 2) produce a
   trustworthy signal — mutating code against a benchmark that can't discriminate
   configs will just add noise.

## Milestone 2: code-level mutation, not just hyperparameters

Once Guardrail 4's benchmark can actually discriminate configs, `propose_mutation`
grows from perturbing `HarnessConfig` fields to proposing a bounded diff to
`agents/planner.py` (e.g., an LLM asked to edit the decomposition rule, given the
current file, its evaluate_variant score, and a size cap on the diff). The
archive/accept/evaluate/split machinery in this document does not change — only the
mutation operator does. New requirements that come with code mutation specifically:

- **Sandboxed execution**: each candidate diff is applied to a throwaway copy of the
  repo (a temp worktree), not the live source tree, and run with a wall-clock
  timeout. A candidate that doesn't finish or doesn't import cleanly is rejected, not
  retried.
- **Regression gate before benchmark gate**: a candidate must pass
  `tests/test_harness_contract.py` (the existing design-constraint tests) before it's
  even eligible for `meta_val` evaluation — cheap filter, catches "technically higher
  score but violates the tool budget" before spending eval budget on it.
- **Diff size cap**: reject proposals touching more than N lines, so a mutation is
  reviewable and its effect attributable, and so the search explores locally
  (small steps) rather than rewriting the planner wholesale each iteration.

## What this is not (yet)

Not weight fine-tuning — nothing here trains a model. Not unconstrained
self-modification — the mutable surface is closed by construction, and the
grader/splits are structurally unreachable. Not a claim of success — see "Current
status" above. This is the scaffold for a legitimate experiment, with the ablation
built in from the start so the eventual claim ("the meta-loop found a harness variant
that generalizes and beats random search") is falsifiable rather than assumed.
