# search-rsi

A harness for **search and discovery agents** where recursive self-improvement (RSI) is
carried entirely by persisted, structured cross-run memory — not by fine-tuning any
model. Sibling of `ouroboros-cv` (vision) and `deliberative_ai` (NLP): same contract
(one executed score, real data end to end, memory-vs-ablation as a first-class run mode),
applied to multi-hop question answering and information discovery.

**The product is the harness, not any single answer it produces.**

- `plan.md` — the build plan: design constraints carried over from the sibling projects,
  layout, first executable slice, and the benchmark that must exist before claiming
  success.
- `docs/ARCHITECTURE.md` — the run loop and memory contract in more detail.
- `docs/META_RSI.md` — the meta layer: a loop that evolves the harness's own
  parameters/code (not just task strategies), with its own guardrails, frozen
  train/meta-val/held-out split, archive, and a random-search ablation arm.

## North star claim

> Given N discovery tasks solved in sequence, task N reaches the target answer-quality
> metric using fewer tool calls than task 1 did on a difficulty-matched task — and beats
> an ablated harness with identical compute and tools but no cross-run memory.

## Layout

- `src/search_rsi/harness/` — plan → retrieve → verify → answer → score → memory-write
  run loop; owns the tool-call budget.
- `src/search_rsi/agents/` — planner (query decomposition) and verifier
  (citation-support check).
- `src/search_rsi/retrieval/` — BM25 and (later) dense retrieval over a real local
  corpus.
- `src/search_rsi/memory/` — persisted strategy store, read/write/promote.
- `src/search_rsi/eval/` — grading, held-out benchmark runner, memory-on-vs-off ablation
  report.
- `benchmarks/corpus/` — small real corpus + gold task set for dev and held-out eval.

## First executable slice

```python
from search_rsi import Harness, Task

harness = Harness.local()  # BM25 retriever + rule-based planner, no network calls
task = Task(
    question="Which document introduces the term first used by the follow-up doc?",
    gold_doc_ids=["doc_02", "doc_07"],
    gold_answer="doc_02",
    tool_budget=6,
)

result = harness.run(task, memory_enabled=True, is_training_task=True)
print(result.answer, result.score, result.tool_calls_used)
```

Run the offline contract tests with `python -m pytest`. Everything above executes real
BM25 retrieval over the committed corpus and real exact-match grading — no mocked
retrieval, no simulated scores (see `plan.md` §2, C1-C2).

## Meta layer: RSI for RSI

`experiments/run_meta_ablation.py` runs a second, outer loop that searches over the
harness's own **retrieval algorithm** (BM25 / Jaccard overlap / TF-IDF cosine — three
different scoring mechanisms, not one algorithm's knobs), BM25's hyperparameters,
the verifier's overlap threshold, and whether generative query expansion (a HyDE-
style stand-in) runs — gated against a frozen `meta_val` task split, with an archive
of accepted variants and a random-search control arm. `experiments/find_production_config.py`
closes the loop end to end: it re-scores the search's Pareto-optimal candidates
(score vs. latency) on held-out eval and exports the winner as a `config.json` +
`serve.py` a production process can load and call directly — "describe a search
problem, get the best-scoring lowest-latency config, deploy it."

See `docs/META_RSI.md` for the guardrails (closed mutation surface, frozen 3-way
split, Pareto-aware acceptance, meta-level ablation) and the current honest result:
the benchmark corpus was rebuilt to have real headroom (decoy documents, length
variation), and the latest run caught a genuine small case of meta-overfitting — an
evolved config that tied the baseline on `meta_val` but scored slightly lower on
`held_out_eval` — exactly what the frozen split exists to catch, reported rather
than hidden.

## Status

Scaffold stage: control-plane contract (Task/TaskResult/Harness/Memory), three real
retrieval algorithms plus generative query expansion, and a meta-loop that searches
over algorithm choice + hyperparameters + generative technique, gated by a frozen
train/meta-val/held-out split, with Pareto selection and a production-export step.
LLM-backed planning, a real embedding-based dense retriever, code-level meta-
mutation, and resolving the current near-tie meta-overfitting case are the next
milestones (see `plan.md` §5, `docs/META_RSI.md` "Current status", "Milestone 2").
