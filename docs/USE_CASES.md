# Search use cases and the three RSI levels

"Search" is not one problem. A product catalog, a codebase, a help center, a
people directory and a news archive each break a default BM25 box in a different
way, and the fix that rescues one (hard price filters, identifier splitting, a
recency prior) is dead weight or actively harmful on another. search-rsi's job is
to **discover the right pipeline for each kind of search problem, and to get better
at that discovery over time**. Nothing is fine-tuned: every improvement is an
inspectable artifact (a rewrite rule, a pipeline config, a problem profile).

## The use-case suite

Every built-in problem is a deterministic generator (`src/search_rsi/problems/`)
with graded relevance judgments and a frozen train / meta_val / held_out_eval split.
Each one isolates a failure mode that real search systems hit:

| problem | use case | what breaks the default | stages that should matter | metric |
|---|---|---|---|---|
| `product_search` | e-commerce catalog | users say "sneakers", catalog says "running shoes"; "under $80" is a constraint, not text | learned rewrites, soft/hard filters, stemming | nDCG@10 |
| `code_search` | NL → code | `parseCfgFile` is one token to a word tokenizer; code abbreviates (`cfg`, `db`, `img`) | `code` analyzer, char n-grams, learned rewrites | MRR@10 |
| `faq_support` | help-center matching | paraphrase ("terminate my membership"), stopword-heavy phrasing, near-duplicate business/personal articles | learned rewrites, fusion, rerank | nDCG@10 |
| `entity_lookup` | navigational name search | typos in rare proper names; near-duplicate names punish fuzzy matching | char n-grams, spell correction, fusion | success@1 |
| `news_freshness` | time-sensitive search | "latest X" matches every yearly edition equally; year-specific queries must not be pulled to recent | recency prior gated on temporal intent | nDCG@10 |
| `multihop_provenance` | multi-hop QA retrieval | decoy docs share vocabulary with the answer | (control: BM25 is already near-ceiling) | nDCG@10 |
| *any BEIR dir* | bring your own | whatever your data does | all of them | nDCG@10 |

`multihop_provenance` is deliberately a control: a problem where the baseline is
already ~0.99 should show the optimizer *not* finding spurious wins.

### Real datasets

`search-rsi optimize path/to/scifact` works on any BEIR-format directory
(`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) — SciFact, NFCorpus, FiQA,
ArguAna, TREC-COVID, or your own logs exported in that shape. Queries are split by a
content hash of their ids, so the split is frozen without modifying the dataset.
Documents may carry a `metadata` object (`price`, `date`, facet strings) and the
filter and recency stages will use it.

### Adding a use case

Write `build() -> SearchProblem` in `src/search_rsi/problems/<name>.py`, register it
in `BUILTIN_PROBLEMS`, and give it a docstring saying what makes it hard and which
stages it should reward. Use `round_robin_splits` grouped by the pattern that has
to generalize (a synonym, an intent), so every split contains every pattern.
`tests/test_problems_and_metrics.py` validates it automatically.

## The pipeline (the thing being improved)

`SearchPipeline` (`src/search_rsi/pipeline/`) runs, in order:

1. **learned rewrites**: expand user vocabulary with corpus vocabulary (memory)
2. **spell correction**: out-of-vocabulary tokens → nearest corpus term
3. **constraint parsing**: price bounds and exact facet values from document metadata
4. **retrieval**: BM25 / TF-IDF / Jaccard / char-trigram / dense (optional), under
   any of four analyzers, with optional title boosting
5. **hybrid fusion**: a second retriever fused by reciprocal rank fusion
6. **pseudo-relevance feedback**: RM3-style expansion from the top documents
7. **filtering**: soft (demote violations) or hard (drop them)
8. **rerank**: query-term coverage
9. **recency prior**: only when the query shows temporal intent

Every stage is switched and tuned by one frozen `PipelineConfig`, the entire
mutable surface. There is no field for the grader, the judgments or the splits.

## Three levels of self-improvement

**Level 1: experience memory (`rsi/learn.py`).** Training queries stream in.
When the pipeline fails one, the learner finds the words the corpus never uses
(excluding typos, which spell correction handles, and price grammar, which the
constraint parser handles) and searches for the corpus words that close the gap,
scored by the real metric. A rule is **promoted** only if replaying earlier
training queries that contain the word shows no regression. It then stays **on
probation**: every later query it fires on is scored with and without it. A rule
that hurts is first **narrowed** (a target dropped) and, if it keeps hurting,
**demoted**. Rules are persisted JSON, e.g. `rucksack → backpack`,
`download → fetch`, `terminate → cancel`.

**Level 2: pipeline evolution (`rsi/optimize.py`).** A DGM-style archive search
over `PipelineConfig`, gated on meta_val. The champion changes only through the
strict Pareto gate (score beyond a significance floor, or equal score for fewer
retrieval calls or less latency). Children that beat their own parent enter the
archive as stepping stones, and parents are picked by tournament. Candidates that
enable learned rewrites get rules learned *with their own pipeline*, so memory and
pipeline co-adapt. A random-search arm with the same budget is always run
alongside.

**Level 3: cross-problem transfer (`rsi/transfer.py`).** Every optimized problem
records its champion and a profile of its shape: document length, identifier
density, structured fields, query gap / typo / temporal rates. A new problem is
profiled from its documents and *train* queries only. The champions of the most
similar known problems then become warm starts for its optimizer, or, with three
train-split probes and no search, the pipeline it ships with.

## Honest status

See `reports/benchmark.md` for the current numbers (3 seeds, held-out eval). The
short version:

- Both search arms beat the hand-set baseline by large margins on held-out eval for
  every problem with headroom, and the control problem shows no spurious gain.
- Learned rewrites alone lift held-out scores substantially where there is a
  vocabulary gap (product, code, FAQ), and removing them from the evolved champion
  shows how much of its win they carry.
- **Evolution does not reliably beat random search at 40 evaluations.** With ~13
  mostly independent stages, random search is a strong baseline (Bergstra & Bengio,
  2012). This is the meta-level ablation doing its job, and the claim "the evolved
  search earns its keep" is not made.
- Transfer with 6 problems is thin: it helps when a genuinely similar problem
  exists and falls back to near-baseline when none does.

## Roadmap: search use cases not covered yet

- **Conversational search**: follow-up queries that depend on earlier turns
  (context carry-over as a pipeline stage).
- **Cross-lingual search**: queries and documents in different languages (learned
  rewrites are a natural fit: a bilingual lexicon mined from failures).
- **Autocomplete / query suggestion**: prefix search, with latency as the primary
  constraint.
- **Geo / local search**: distance as a constraint and a ranking feature.
- **Personalized and permission-aware enterprise search**: per-user filters and
  boosts.
- **Agentic research**: the original multi-hop `Harness` with an LLM planner, a
  tool budget, and this pipeline as its search tool.
- **Dense retrieval at scale**: plug in `sentence_transformers_embedder()` and an
  ANN index; the optimizer already treats `dense` as one more retriever.
- **LLM-proposed mutations** (docs/META_RSI.md Milestone 2): code-level pipeline
  stages proposed by a model, under the same gate and sandbox.
