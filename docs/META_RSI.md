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
(`src/search_rsi/meta/config.py`): which retrieval **algorithm** runs (`retriever`:
`bm25` / `jaccard` / `tfidf` — three genuinely different scoring mechanisms, not one
algorithm's knobs, see "Algorithm selection, not just tuning" below), BM25's
`k1`/`b` when BM25 is selected, the verifier's `min_overlap`, and whether generative
query expansion (`use_hyde`) runs on top of whichever algorithm was picked. There is
no field for the grader, the gold task set, or the tool-budget accounting — a system
that can edit its own exam will learn to edit its exam, not to search better.
Milestone 2 (below) grows this surface to short code diffs in `agents/planner.py`
and to new retrieval algorithm *implementations* (not just a choice among existing
ones), but the same principle holds: the diff target is never `eval/grading.py`,
`benchmarks/corpus/gold_tasks.json`, or the split boundaries.

## Algorithm selection, not just tuning

The point of this system isn't "tune BM25's two knobs." Three retrieval mechanisms
with genuinely different scoring math are registered
(`src/search_rsi/retrieval/registry.py`):

- **BM25** (`bm25.py`) — term-frequency saturation + length normalization (`b`) +
  IDF. The default choice for keyword search; `b` trades off penalizing long
  documents.
- **Jaccard overlap** (`jaccard.py`) — pure set overlap, no frequency or length
  weighting at all. Wins when documents are short and roughly uniform in length,
  where BM25's length correction is dead weight.
- **TF-IDF cosine** (`tfidf.py`) — unit-normalized vectors, no saturation term.
  Rewards proportional term overlap but a single rare shared term can dominate the
  normalized vector — the benchmark corpus's decoy documents are specifically built
  to expose this (see `tests/test_retrieval_algorithms.py`, which found TF-IDF
  latching onto a decoy document that BM25 and Jaccard both ignored).

Orthogonal to algorithm choice: **generative query expansion** (`use_hyde`,
`retrieval/hyde.py`) — a deterministic stand-in for the generation step in HyDE
(Gao et al., 2022): instead of retrieving with the terse query, expand it into a
richer pseudo-document first, closing the lexical gap between how a question is
phrased and how its answer document is written. It plays the same role
`RuleBasedPlanner` plays for query decomposition — a fixed template stands in for an
LLM call so the wrapper's contract is testable offline; swapping in a real LLM call
is a one-function change (see Milestone 2's "true generative retrieval" note for
what's still out of scope even after that swap).

Adding a fourth algorithm (e.g. a real embedding-based dense retriever, which needs
model weights and breaks the current CPU-only/dependency-free constraint) means
adding one branch to `build_retriever` and one name to `RETRIEVER_NAMES` — nothing
in the harness, evaluator, or meta-loop changes.

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

## The actual product loop: latency is a first-class output, not an afterthought

The deliverable is not "the config with the highest score." It's: given a search
problem (a corpus, a benchmarked task set, a latency budget), find the
best-scoring, lowest-latency configuration and hand back something a production
server can load — not a benchmark table someone has to translate into a deploy
decision by hand. Concretely:

- `evaluate_variant` measures **mean latency per query** (`mean_latency_ms`)
  alongside score and tool calls, timed end-to-end per task with index-build
  amortized outside the loop (a warm server, not a cold start).
- `Archive.pareto_front()` returns the non-dominated set on (higher score, lower
  latency) — a real shortlist (a slower-but-more-accurate option next to a
  faster-but-slightly-worse one), not one number that hides the tradeoff from
  whoever has to deploy it.
- `experiments/find_production_config.py` runs the meta-loop, re-scores every
  Pareto-front candidate on **held-out eval** (never on `meta_val` — the split
  discipline applies to the deployment decision too), picks the best-scoring
  option among them (ties broken by lower latency), and calls
  `search_rsi.meta.export_variant` to write `config.json` + a `serve.py` that
  loads it and exposes a plain `answer(harness, question)` call. This was smoke-
  tested end to end: the exported artifact answers a real held-out-style question
  correctly with its own process, no reference back to the meta-loop's internals.

## Current status (honest)

The corpus was regenerated (`benchmarks/corpus/generate_corpus.py`: 12 topic chains,
48 documents, 24 gold tasks, 8/8/8 train/meta-val/held-out) specifically to break the
v1 ceiling effect — short uniform documents let BM25's `k1`/`b` decide nothing, and
`min_overlap` never bound. It now includes decoy documents sharing surface
vocabulary with the correct answer (to make `min_overlap` matter) and length
variation (to make `b` matter), plus three genuinely different algorithms and
generative expansion as real, exercised options rather than one algorithm's dials.

Running `experiments/run_meta_ablation.py` now shows **non-zero acceptance on both
arms** (typically 1-3/30) — real signal where before there was none. But the honest
result so far is a **meta-overfitting catch, not a win**: the evolved arm's champion
(`jaccard[min_overlap=2]`, tied with baseline's 1.0 on `meta_val`) scores *lower*
than the untouched baseline on `held_out_eval` (0.981 vs. 1.000). This is exactly
what the frozen 3-way split exists to catch — an 8-task `meta_val` split is still
small enough that a tie on it doesn't guarantee generalization — and it's reported
here rather than cherry-picking a seed that looks better.

**What would close this gap:**
1. More meta-val tasks (8 is workable but still small for trusting a tie-break) —
   the generator makes this cheap: raise `N` topic chains in
   `generate_corpus.py` and the 8/8/8 split scales with it automatically.
2. A stricter acceptance rule for near-ties: currently `tied_score_cheaper` accepts
   on *any* latency or tool-call improvement at equal score; requiring the tie to
   also hold (or improve) on held-out-eval-shaped validation tasks distinct from
   `meta_val` (a 4-way split: train / meta-val / meta-test / held-out-eval) would
   catch this specific failure before deployment, at the cost of needing yet more
   tasks.
3. Only once ties are resolved reliably does growing the mutation surface to code
   diffs (Milestone 2) produce a trustworthy signal — mutating code against a
   benchmark that can't reliably rank near-ties will just add noise on top of noise.

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

**Out of scope even after Milestone 2, and why:** "true" generative retrieval —
e.g. a Differentiable Search Index (Tay et al., 2022) that trains a model to emit
document identifiers directly, or a real (non-templated) HyDE that calls an actual
LLM — requires either training a model or an external API call. Both break this
project's CPU-only, dependency-free, no-fine-tuning constraint (`plan.md` C2, C3).
They're legitimate future directions, but as a separate track with its own
compute/cost accounting, not folded into the meta-loop's evaluation budget where an
API call's latency and cost would silently dominate every comparison.

## What this is not (yet)

Not weight fine-tuning — nothing here trains a model. Not unconstrained
self-modification — the mutable surface is closed by construction, and the
grader/splits are structurally unreachable. Not a claim of success — see "Current
status" above. This is the scaffold for a legitimate experiment, with the ablation
built in from the start so the eventual claim ("the meta-loop found a harness variant
that generalizes and beats random search") is falsifiable rather than assumed.
