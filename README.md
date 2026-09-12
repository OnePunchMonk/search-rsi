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

## Status

Scaffold stage: control-plane contract (Task/TaskResult/Harness/Memory) and the local
BM25 + rule-based-planner slice. LLM-backed planning, dense retrieval, and the
N-task ablation benchmark are the next milestones (see `plan.md` §5).
