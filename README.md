# search-rsi

**Recursive self-improvement for search.** Give it a search problem (a corpus plus
judged queries) and it works out which pipeline fits: analyzer, retriever, hybrid
fusion, query rewriting, filters, reranking, recency. It learns query rewrites from
its own failures and carries what worked over to similar problems. Nothing is
fine-tuned. Every improvement is an inspectable artifact that can be re-verified: a
rewrite rule, a pipeline config, a problem profile.

```
$ search-rsi search product_search 'red sneakers under $90'        # baseline BM25
  4.0191  p043   Corvid Core 9 Rain Jacket - Grey                    # wrong category
$ search-rsi learn product_search                                  # learn from train queries
  sneakers -> shoe mesh    trainers -> running    rucksack -> backpack   ...
$ search-rsi search product_search 'red sneakers under $90' \
    --config '{"use_learned_rewrites": true, "filter_mode": "soft"}'
  constraints: price_max=90, color=red
  6.4244  p015   Fjell Flex 6 Running Shoes - Red
$ search-rsi optimize code_search --export deploy/code             # evolve the whole pipeline
  held-out mrr@10: baseline 0.480 -> champion 0.786
```

## Search use cases

Six built-in problems, each isolating one way a default search box fails. Any
BEIR-format dataset (SciFact, FiQA, NFCorpus, your own logs) loads by path.

| problem | what breaks the default |
|---|---|
| `product_search` | "sneakers" vs. "running shoes"; "under $80" is a constraint, not text |
| `code_search` | `parseCfgFile` is one opaque token; code abbreviates what people spell out |
| `faq_support` | paraphrase, stopword-heavy phrasing, near-duplicate articles |
| `entity_lookup` | typos in rare names; near-duplicate names punish fuzzy matching |
| `news_freshness` | "latest X" needs a recency prior, and "X 2019" must not get one |
| `multihop_provenance` | control: the baseline is already near ceiling |

See **[docs/USE_CASES.md](docs/USE_CASES.md)** for the full matrix, the pipeline
stages, how to add a use case, and the roadmap (conversational, cross-lingual,
autocomplete, geo, enterprise, agentic research).

## Three levels of self-improvement

1. **Experience memory** (`rsi/learn.py`): query-rewrite rules mined from failed
   *training* queries. A rule is promoted only if replaying earlier queries shows no
   regression. After that it stays on probation, and is narrowed or demoted if it
   starts to hurt.
2. **Pipeline evolution** (`rsi/optimize.py`): a DGM-style archive search over the
   whole pipeline config. It is gated on a frozen meta_val split and always runs
   next to a random-search control with the same budget. Rules are re-learned for
   each candidate pipeline, so memory and pipeline co-adapt.
3. **Cross-problem transfer** (`rsi/transfer.py`): each champion is stored with a
   profile of its problem's shape, and warm-starts new problems that look like it.

## Results (held-out eval, 3 seeds, 40 evaluations per arm)

| problem | metric | baseline | + memory | evolved | random search |
|---|---|---|---|---|---|
| product_search | nDCG@10 | 0.609 | 0.822 | 0.894 ± 0.030 | 0.905 ± 0.011 |
| code_search | MRR@10 | 0.480 | 0.613 | 0.858 ± 0.015 | 0.863 ± 0.072 |
| faq_support | nDCG@10 | 0.425 | 0.582 | 0.666 ± 0.021 | 0.666 ± 0.026 |
| entity_lookup | success@1 | 0.419 | 0.419 | 1.000 ± 0.000 | 0.989 ± 0.015 |
| news_freshness | nDCG@10 | 0.572 | 0.572 | 0.871 ± 0.183 | 1.000 ± 0.000 |
| multihop_provenance | nDCG@10 | 0.986 | 0.986 | 0.986 ± 0.000 | 0.986 ± 0.000 |

What this shows, and what it doesn't. Full report: [reports/benchmark.md](reports/benchmark.md).

- **Per-use-case pipeline search works.** Both search arms beat the hand-set
  baseline by large margins wherever there is headroom, and they find no
  spurious gain on the control problem.
- **Learned memory generalizes.** Rules learned on train lift *held-out* queries
  by 0.13–0.21 where there is a vocabulary gap. With rewrites removed from the
  evolved champion, product search falls from 0.894 to 0.674.
- **Evolution does not beat random search yet.** At this budget they are
  statistically tied, and on one seed evolution never found the recency stage for
  news. The meta-level ablation exists to catch exactly this, so no claim that
  "the evolved search earns its keep" is made.
- **Transfer is thin with 6 problems.** It helps when a similar problem exists and
  gives little when none does.

## Quick start

```bash
pip install -e ".[dev]"          # pure Python, no dependencies; [dense] adds sentence-transformers
python -m pytest                 # 41 tests, ~3s, fully offline
search-rsi problems
search-rsi optimize code_search --export deploy/code
python deploy/code/serve.py corpus.jsonl "how to download an http request"
search-rsi optimize path/to/beir/scifact     # any BEIR-format directory
search-rsi bench --out reports/              # full benchmark, ~4 min
```

The exported directory (`config.json`, `rewrites.json`, `manifest.json`, `serve.py`)
is the deliverable: a pipeline plus its learned memory, loadable by a service, with
held-out numbers recorded in the manifest.

## Guardrails (from the original design: plan.md, docs/META_RSI.md)

- **One executed score** per problem, computed against real judgments.
  `eval/metrics.py` is unreachable from the mutation surface.
- **Frozen three-way split.** Train feeds memory, meta_val gates every accept/reject
  decision, and held-out eval is only reported. Tests check that the optimizer never
  reads held-out and that evaluation never writes memory.
- **Every RSI level has a control arm**: memory on vs. off, evolution vs. random
  search, transfer vs. baseline.
- **Offline and deterministic**: real retrieval over real text, with no mocks and no
  network. A dense embedder is an optional plug-in
  (`retrieval.set_embedder(sentence_transformers_embedder())`).

## Layout

- `src/search_rsi/problems/`: use-case generators, `SearchProblem`, BEIR loader
- `src/search_rsi/pipeline/`: `PipelineConfig`, `SearchPipeline`, query understanding
- `src/search_rsi/retrieval/`: BM25, TF-IDF, Jaccard, char-trigram, dense, RRF fusion
- `src/search_rsi/text.py`: analyzers (plain, stem, stem+stopwords, code identifiers)
- `src/search_rsi/eval/metrics.py`: the fixed grader
- `src/search_rsi/rsi/`: learn, optimize, transfer, export, bench
- `src/search_rsi/cli.py`: the `search-rsi` command
- `src/search_rsi/harness/`, `agents/`, `meta/`: the original agentic multi-hop
  harness (tool budget, planner/verifier, strategy memory) and its meta-loop, kept
  intact. It is the planned consumer of this pipeline as its search tool (see the
  roadmap in docs/USE_CASES.md).
