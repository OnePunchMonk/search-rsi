# Architecture

## Run loop

```
Task
  -> Planner.plan_queries(task, memory)      # reads memory, does not write it
  -> for each query (bounded by tool_budget):
       Retriever.search(query)               # real BM25 over real corpus
       Verifier.supports(question, doc)      # rejects unsupported hits
  -> answer = top supported doc
  -> score = eval.exact_match_score(task, answer, citations)   # the one executed score
  -> if training task and score < 1.0: Memory.write(strategy)  # never on eval tasks
-> TaskResult
```

## Memory contract

`MemoryStore` is a flat JSON file keyed by `(task_type, domain)`. It is read at the top
of every run (when `memory_enabled=True`) and injected into the planner as strategy
hints — plain text, not weights. It is written only when `is_training_task=True`. Eval
runs (`is_training_task=False`) must leave the store byte-for-byte unchanged; this is
enforced by `tests/test_harness_contract.py::test_holdout_eval_never_writes_memory`.

## Why a rule-based planner first

The planner interface (`plan_queries(task, memory) -> list[str]`) is the seam where an
LLM planner will later be swapped in. Keeping the first implementation deterministic and
dependency-free means the harness contract (budget enforcement, memory read/write
timing, scoring) is fully testable in CI without network access or API keys, before any
model dependency is introduced. This mirrors `self-improving-diffusion`'s local
deterministic backend and `ouroboros-cv`'s CPU-only offline mode.

## What "self-improving" means here

Nothing about the planner, retriever, or verifier is retrained. What changes across
tasks is the *content* of `memory/store/entries.json`: as training tasks reveal that a
single-query plan under-performs on multi-part questions, a strategy entry is recorded,
and the planner reads it back on the next matching task to decompose the question before
retrieving. Improvement is measured, not assumed — see `plan.md` §7 for the ablation
benchmark that must show separation between the memory-on and memory-off curves.
