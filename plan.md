# Build Plan: `search-rsi` — A Self-Improving Search & Discovery Harness

> Sibling of `ouroboros-cv` (vision) and `deliberative_ai` (NLP). Reuses the same
> self-improvement contract — persisted cross-run memory, one executed score, hard
> ablation requirement — applied to agentic search and discovery: multi-hop question
> answering, literature/entity discovery, and tool-mediated information gathering.

---

## 1. What this is

An agent that takes a natural-language information need ("find the three papers that
first proposed X and show how the claim changed across them") plus access to a corpus
and a small toolset (lexical search, dense search, fetch/read, optional web search), and
iteratively plans queries, retrieves, verifies, and answers — with **no fine-tuning of
the underlying model**. What improves across tasks is a persisted, structured memory of
query strategies, tool-use policies, and failure diagnostics that the planner conditions
on for the next task.

**The product is the harness, not any single answer it produces.** Success is that the
harness gets measurably better at search/discovery tasks *as a harness*, across tasks,
without retraining anything.

### North star claim to defend

> Given N discovery tasks solved in sequence, task N reaches the target answer-quality
> metric using fewer tool calls (or lower latency/cost) than task 1 did on a
> difficulty-matched task — and beats an ablated harness with identical compute and
> tools but no cross-run memory.

If the benchmark in §7 can't demonstrate this with a real control arm, the project has
failed regardless of code quality.

---

## 2. Design constraints (carried over from the sibling projects — do not repeat their failures)

### C1. There is exactly one score, and it is executed, not estimated

The only number that drives any decision (accept/reject a plan, promote a strategy to
memory, stop iterating) is computed by actually running the retrieval/answer pipeline
against real documents and a real grader (exact-match / supported-citation check /
gold-passage overlap). No `base_score + heuristic_bonus`. No LLM self-grading standing in
for a checkable ground truth when one exists.

**Test requirement:** corrupt the gold answers or the corpus for one task and assert the
score changes. A data-independent score fails this test.

### C2. Dev/offline mode runs real retrieval on a real (tiny) corpus — never simulated

The fast local path must still index and query real text (a few dozen small documents
committed under `benchmarks/corpus/`), not mock/stub retrieval results. Deterministic and
CPU-only, but the pipeline under test is the real one, wired end to end.

### C3. Cross-run improvement is carried by persisted memory, never by fine-tuned weights

Between-task learning is stored as structured, inspectable artifacts in
`src/search_rsi/memory/` — e.g. "for multi-hop entity questions, decompose into
sub-queries before dense retrieval" or "this domain's lexical search over-matches on
acronyms, prefer dense search first." These are retrieved and injected into the planner's
context for the next task. No gradient updates, no weight checkpoints.

### C4. Eval tasks are held out and never enter memory

Memory is written only from tasks flagged as "training"/practice tasks. Held-out eval
tasks may read memory but must never write to it, even on failure. A test asserts memory
file mtimes/hashes are unchanged after running the eval suite.

### C5. Tool-call budget is part of the score, not a side metric

Every task has a hard cap on tool calls (search/fetch invocations). Exceeding it is a
failure, not a slow success. This is what makes "fewer tool calls at equal quality"
measurable and prevents the agent from brute-forcing quality via unlimited search.

### C6. The ablation control arm is a first-class run mode, not an afterthought

`search_rsi.harness.run(..., memory_enabled=False)` must exist and be exercised by the
benchmark script from day one — not bolted on once the memory-enabled path looks good.

---

## 3. Layout

- `src/search_rsi/harness/` — the run loop: plan → retrieve → verify → answer → score →
  (if training task) write memory. Owns the tool-call budget and stop conditions.
- `src/search_rsi/agents/` — planner (decomposes the information need into a query
  program) and verifier (checks retrieved evidence actually supports a claim before the
  answerer commits to it).
- `src/search_rsi/retrieval/` — lexical (BM25-style) and dense retrieval backends over
  `benchmarks/corpus/`; both are real, small, dependency-light implementations for the
  offline slice.
- `src/search_rsi/memory/` — the persisted strategy store: read/write/promote of
  structured memory entries, keyed by task-feature signature (task type, domain, failure
  mode observed).
- `src/search_rsi/eval/` — scoring (exact-match / supported-citation grading against gold
  answers), the held-out benchmark runner, and the ablation report (memory on vs. off).
- `benchmarks/corpus/` — small real text corpus + a gold task set (question, supporting
  doc ids, expected answer) for both dev iteration and held-out eval.
- `experiments/` — run logs and configs per benchmark sweep.

---

## 4. First executable slice

1. A typed `Task` (question, tool budget, corpus handle) and `TaskResult` (answer,
   citations, tool calls used, score).
2. A local BM25 retriever over `benchmarks/corpus/` — no external services required.
3. A deterministic planner stub (rule-based query decomposition) standing in for an LLM
   planner, so the control-plane contract (plan → retrieve → verify → answer → score →
   memory-write) is testable without API calls, exactly as the diffusion project's local
   deterministic backend stands in for a real denoiser.
4. `search_rsi.harness.run(task, memory_enabled=True|False)` wired end to end against
   (2) and (3), scored by (real) exact-match against the gold task set.
5. Memory store as flat JSON under `memory/store/`, read/write through a small typed API
   so it's trivial to swap the backing store later.

Swapping the rule-based planner for a real LLM planner is a later milestone and should
require no change to the harness contract — only a new `Planner` implementation.

---

## 5. Milestones after the first slice

1. Add a real embedding-based dense retriever as a second backend; benchmark BM25 vs.
   dense vs. hybrid on the same gold set.
2. Replace the rule-based planner with an LLM planner (behind the same `Planner`
   interface) that reads injected memory entries as few-shot context.
3. Add a verifier stage that rejects unsupported citations and forces a re-query,
   counted against the tool budget (tests C1/C5 interaction).
4. Run the N-task sequential benchmark (§7) with memory on vs. off and publish the curve.

---

## 6. Benchmark corpus

Start with ~30-50 short documents spanning a handful of entities/topics with genuine
multi-hop structure (answering some questions requires combining 2+ documents), plus a
gold task file. Synthetic but internally consistent is fine for the offline slice; a
larger real corpus (e.g. a Wikipedia slice) is a later milestone, gated on the offline
contract passing first.

---

## 7. The benchmark that must exist before claiming success

Run the same ordered sequence of N tasks twice: once with `memory_enabled=True`, once
with `memory_enabled=False`, same model/tools/budget. Plot metric-per-task and
tool-calls-per-task for both arms. The claim in §1 holds only if the memory arm's curve
improves over the sequence and separates from the flat/no-improvement ablation curve.

---

## 8. v2: from one benchmark to the search use-case suite

The slice above proved the contract on one multi-hop corpus. v2 generalizes it to
"everything search":

- **Problems, not a corpus**: `SearchProblem` (docs + graded qrels + frozen split)
  with six built-in use cases (product, code, FAQ, entity/typo, news/freshness, the
  original multi-hop set as a control) and a BEIR loader for real datasets.
- **Standard IR metrics** as the executed score (nDCG@10, MRR@10, success@1).
- **A full pipeline as the mutable surface**: analyzers, five retrievers, RRF
  fusion, spell correction, PRF, structured filters, rerank, recency.
- **Memory that actually learns**: query-rewrite rules mined from failed training
  queries with promotion, probation and demotion (replacing the single hard-coded
  "decompose" strategy for this stack).
- **Three RSI levels, each with a control arm**: memory on/off, evolution vs.
  random search, cross-problem transfer vs. baseline.

Status and numbers: `reports/benchmark.md`; design: `docs/USE_CASES.md`.
