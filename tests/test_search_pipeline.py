from dataclasses import replace

from search_rsi.memory import RewriteRule, RewriteRules
from search_rsi.pipeline import BASELINE_PIPELINE, PipelineConfig, SearchPipeline
from search_rsi.pipeline.query import SpellCorrector, parse_constraints
from search_rsi.problems import load_problem
from search_rsi.retrieval import DenseIndex, reciprocal_rank_fusion
from search_rsi.text import edit_distance, get_analyzer, light_stem, split_identifiers
from search_rsi.types import Document


def test_analyzers():
    assert split_identifiers("parseCfgFile") == ["parse", "cfg", "file", "parsecfgfile"]
    assert split_identifiers("load_env_vars") == ["load", "env", "vars"]
    assert light_stem("shoes") == light_stem("shoe")
    assert light_stem("parsing") == light_stem("parse")
    assert "the" not in get_analyzer("stem_stop")("the running shoes")
    assert edit_distance("kellington", "kelingtno") == 2


def test_code_analyzer_is_what_matches_camel_case():
    docs = [Document("a", "function parseCfgFile(path) {}"), Document("b", "function saveUser(u) {}")]
    plain = SearchPipeline(docs, BASELINE_PIPELINE)
    code = SearchPipeline(docs, replace(BASELINE_PIPELINE, analyzer="code"))
    assert plain.search("parse cfg file").doc_ids == []
    assert code.search("parse cfg file").doc_ids[0] == "a"


def test_charngram_and_spell_correction_recover_typos():
    problem = load_problem("entity_lookup")
    target = problem.documents[0]
    typo = target.title.lower()[:3] + target.title.lower()[4:]  # drop one letter
    for config in (PipelineConfig(retriever="charngram"), PipelineConfig(spell_correct=True)):
        top = SearchPipeline(problem.documents, config, cache=problem.index_cache).search(typo).doc_ids[:1]
        assert top == [target.doc_id], config.label()
    assert SpellCorrector(problem.documents).correct_token("sneakerzz") == "sneakerzz"  # nothing close


def test_constraint_parsing_and_soft_vs_hard_filters():
    facets = {"color": {"red", "blue"}, "category": {"running shoes", "backpack"}}
    text, c = parse_constraints("red running shoes under $80", facets)
    assert c.price_max == 80 and c.facets == {"color": "red", "category": "running shoes"}
    assert "80" not in text

    docs = [
        Document("cheap_blue", "blue running shoes", metadata={"color": "blue", "price": 50}),
        Document("red_ok", "red running shoes", metadata={"color": "red", "price": 70}),
        Document("red_pricey", "red running shoes shoes", metadata={"color": "red", "price": 120}),
    ]
    soft = SearchPipeline(docs, PipelineConfig(filter_mode="soft")).search("red running shoes under $80")
    hard = SearchPipeline(docs, PipelineConfig(filter_mode="hard")).search("red running shoes under $80")
    assert soft.doc_ids[0] == "red_ok" and len(soft.doc_ids) == 3
    assert hard.doc_ids == ["red_ok"]


def test_recency_prior_only_fires_on_temporal_intent():
    problem = load_problem("news_freshness")
    pipe = SearchPipeline(problem.documents, PipelineConfig(recency_weight=0.4), cache=problem.index_cache)
    latest = pipe.search("latest wind energy outlook").doc_ids[0]
    newest = max((d for d in problem.documents if d.metadata["topic"] == "wind energy"),
                 key=lambda d: d.metadata["date"])
    assert latest == newest.doc_id
    assert pipe.search("wind energy outlook 2015").doc_ids[0] == "rep_wind_energy_2015"


def test_rewrite_rules_expand_without_replacing():
    rules = RewriteRules([RewriteRule("sneakers", ("running", "shoes"), 1, 0.5, "")])
    assert rules.apply("red sneakers") == "red sneakers running shoes"
    assert rules.apply("red sneaker") == "red sneaker running shoes"  # stem match
    assert RewriteRules.from_json(rules.to_json()).apply("sneakers") == "sneakers running shoes"


def test_fusion_and_dense_backend():
    fused = reciprocal_rank_fusion([[("a", 9.0), ("b", 1.0)], [("b", 0.9), ("c", 0.1)]], k=60)
    assert fused[0][0] == "b"  # ranked by both retrievers

    docs = [Document("x", "alpha"), Document("y", "beta")]
    embed = lambda texts: [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]  # noqa: E731
    assert DenseIndex(docs, embed).search("alpha please", 1)[0][0] == "x"
