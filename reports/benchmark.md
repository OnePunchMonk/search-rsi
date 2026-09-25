# search-rsi benchmark

Optimizer budget: 40 evaluations per arm; optimizer columns are the mean ± std over seeds [0, 1, 2]. Every score is on **held-out eval** unless marked meta_val: held-out queries never influenced a rule, an accept/reject decision, or a champion.

## Headline

| problem | metric | baseline | + memory | evolved | evolved, memory off | random search | transfer (3 probes) |
|---|---|---|---|---|---|---|---|
| product_search | ndcg@10 | 0.609 | 0.822 | **0.894 ± 0.030** | 0.674 ± 0.050 | 0.905 ± 0.011 | 0.845 |
| code_search | mrr@10 | 0.480 | 0.613 | **0.858 ± 0.015** | 0.807 ± 0.015 | 0.863 ± 0.072 | 0.868 |
| faq_support | ndcg@10 | 0.425 | 0.582 | **0.666 ± 0.021** | 0.439 ± 0.038 | 0.666 ± 0.026 | 0.680 |
| entity_lookup | success@1 | 0.419 | 0.419 | **1.000 ± 0.000** | 1.000 ± 0.000 | 0.989 ± 0.015 | 1.000 |
| news_freshness | ndcg@10 | 0.572 | 0.572 | **0.871 ± 0.183** | 0.871 ± 0.183 | 1.000 ± 0.000 | 0.612 |
| multihop_provenance | ndcg@10 | 0.986 | 0.986 | **0.986 ± 0.000** | 0.986 ± 0.000 | 0.986 ± 0.000 | 0.962 |

## Champions (chosen on meta_val, seed 0)

- **product_search**: `charngram[spell_correct=True,filter_mode=soft,use_learned_rewrites=True]` (meta_val 0.952, accepted 9/40; random arm `bm25+charngram[bm25_k1=1.67,bm25_b=0.84,rrf_k=10,title_boost=1,spell_correct=True,prf_terms=5,filter_mode=soft,recency_weight=0.60,rerank_weight=0.20,use_learned_rewrites=True]` meta_val 0.952); latency 0.06 → 0.81 ms/query
  - transfer: shipped `charngram+bm25[use_learned_rewrites=True]` from faq_support (profile distance 1.47; probed faq_support, code_search, multihop_provenance)
- **code_search**: `charngram[use_learned_rewrites=True]` (meta_val 0.844, accepted 8/40; random arm `tfidf+charngram[analyzer=stem_stop,rrf_k=100,title_boost=1,filter_mode=hard,rerank_weight=0.20,use_learned_rewrites=True]` meta_val 0.740); latency 0.02 → 0.15 ms/query
  - transfer: shipped `charngram[spell_correct=True,filter_mode=soft,use_learned_rewrites=True]` from product_search (profile distance 2.40; probed faq_support, entity_lookup, product_search)
- **faq_support**: `charngram+bm25[use_learned_rewrites=True]` (meta_val 0.756, accepted 5/40; random arm `tfidf+jaccard[analyzer=code,rrf_k=10,recency_weight=0.20,rerank_weight=0.20,use_learned_rewrites=True]` meta_val 0.816); latency 0.04 → 0.18 ms/query
  - transfer: shipped `charngram[spell_correct=True,filter_mode=soft,use_learned_rewrites=True]` from product_search (profile distance 1.47; probed entity_lookup, product_search, code_search)
- **entity_lookup**: `charngram` (meta_val 1.000, accepted 2/40; random arm `charngram+tfidf[analyzer=code,rrf_k=10,filter_mode=soft,recency_weight=0.40]` meta_val 1.000); latency 0.01 → 0.14 ms/query
  - transfer: shipped `charngram[use_learned_rewrites=True]` from code_search (profile distance 2.34; probed faq_support, code_search, product_search)
- **news_freshness**: `tfidf` (meta_val 0.644, accepted 6/40; random arm `tfidf+jaccard[analyzer=stem_stop,rrf_k=100,title_boost=3,filter_mode=soft,recency_weight=0.60]` meta_val 1.000); latency 0.15 → 0.11 ms/query
  - transfer: shipped `charngram[spell_correct=True,filter_mode=soft,use_learned_rewrites=True]` from product_search (profile distance 2.79; probed faq_support, product_search, multihop_provenance)
- **multihop_provenance**: `bm25` (meta_val 0.991, accepted 0/40; random arm `bm25` meta_val 0.991); latency 0.06 → 0.12 ms/query
  - transfer: shipped `charngram[spell_correct=True,filter_mode=soft,use_learned_rewrites=True]` from product_search (profile distance 2.45; probed faq_support, product_search, code_search)

## Learned rewrite memory (baseline pipeline, train split)

- **product_search**: 15 rules (0 demoted on probation, 0 rejected at promotion); second half of the train stream: 0.718 with memory vs. 0.718 without. `canteen→hydration/water`, `daypack→commuting/backpack`, `earbuds→headphones/travel`, `exercise→yoga/pilates`, `flask→water`, `mountaineering→terrain/trails`, `raincoat→rain/heavy`, `reading→lamp`
- **code_search**: 6 rules (0 demoted on probation, 1 rejected at promotion); second half of the train stream: 0.557 with memory vs. 0.365 without. `database→conn`, `download→get/client`, `image→write/img`, `interpret→tokens/raw`, `persist→write`, `timestamp→reader/read`
- **faq_support**: 9 rules (2 demoted on probation, 4 rejected at promotion); second half of the train stream: 0.436 with memory vs. 0.385 without. `authenticate→verify/email`, `confirm→verify`, `erase→payment/delete`, `modify→update`, `passcode→password`, `possible→shipping`, `purchase→order`, `retrieve→history`
- **entity_lookup**: 1 rules (0 demoted on probation, 0 rejected at promotion); second half of the train stream: 0.133 with memory vs. 0.133 without. `founded→glaciers`
- **news_freshness**: 0 rules (0 demoted on probation, 0 rejected at promotion); second half of the train stream: 0.585 with memory vs. 0.585 without. 
- **multihop_provenance**: 1 rules (0 demoted on probation, 2 rejected at promotion); second half of the train stream: 0.982 with memory vs. 0.982 without. `redefinition→hop`
