from search_rsi.benchmarks_corpus import load_default_corpus
from search_rsi.retrieval import RETRIEVER_NAMES, build_retriever


def test_every_registered_algorithm_returns_real_scored_hits():
    """Not a benchmark result — the corpus deliberately includes decoy documents
    that some algorithms are more vulnerable to than others (see tfidf.py's
    documented rare-term-dominance tradeoff); which algorithm wins on which task
    shape is the meta-loop's job to discover, not something to hardcode here. This
    is just a sanity floor: every registered algorithm must be a real, working
    implementation over real data, not a stub that returns nothing."""
    docs = load_default_corpus()
    for name in RETRIEVER_NAMES:
        retriever = build_retriever(name, docs)
        hits = retriever.search("term latency budget origin protocol", top_k=3)
        assert hits, f"{name} returned no hits at all"
        assert all(score > 0 for _, score in hits), f"{name} returned a non-positive score"


def test_hyde_expansion_actually_changes_the_query():
    from search_rsi.retrieval.hyde import hyde_expand

    original = "origin of the term latency budget"
    expanded = hyde_expand(original)
    assert expanded != original
    assert original in expanded


def test_algorithm_choice_is_part_of_the_meta_mutation_space():
    """docs/META_RSI.md: the meta-loop must be able to select a different retrieval
    *mechanism*, not just tune one algorithm's hyperparameters."""
    import random

    from search_rsi.meta.config import HarnessConfig
    from search_rsi.meta.mutate import propose_mutation

    rng = random.Random(0)
    parent = HarnessConfig(retriever="bm25")
    seen_retrievers = set()
    for _ in range(200):
        seen_retrievers.add(propose_mutation(parent, rng).retriever)
    assert seen_retrievers == set(RETRIEVER_NAMES)
