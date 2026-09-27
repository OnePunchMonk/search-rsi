"""The inner RSI loop for search pipelines: learn query rewrites from experience.

Training queries are processed in order, like a stream of user searches with
relevance feedback. When the pipeline fails one, the learner looks for a
vocabulary gap — a query word the corpus never uses — and searches for the corpus
words that close it, scored by the executed metric on that query. A candidate rule
is promoted only if replaying every earlier training query that contains the same
word shows no regression, and every promoted rule stays on probation: each later
training query it fires on is scored with and without it, and a rule that has
hurt at least as often as it has helped is demoted. Held-out queries never pass
through here (plan.md C4).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace

from search_rsi.eval.metrics import METRICS
from search_rsi.memory.rewrites import RewriteRule, RewriteRules
from search_rsi.pipeline import PipelineConfig, SearchPipeline
from search_rsi.pipeline.query import SpellCorrector, parse_constraints
from search_rsi.problems.base import Query, SearchProblem
from search_rsi.text import STOPWORDS, light_stem, split_identifiers, tokenize

_MAX_SOURCES = 3
_MAX_TARGET_CANDIDATES = 8
_MAX_TARGETS = 2


@dataclass
class LearningTrace:
    rules: RewriteRules
    online_scores: list[float] = field(default_factory=list)  # score when each query arrived
    promoted: list[RewriteRule] = field(default_factory=list)
    demoted: list[RewriteRule] = field(default_factory=list)
    rejected: int = 0  # candidates that helped their query but regressed replay


class _Vocab:
    def __init__(self, problem: SearchProblem):
        cached = problem.index_cache.get(("learn_vocab",))
        if cached is None:
            df: Counter[str] = Counter()
            stems: set[str] = set()
            for d in problem.documents:
                toks = set(split_identifiers(f"{d.title} {d.text}"))
                df.update(toks)
                stems |= {light_stem(t) for t in toks}
            cached = problem.index_cache[("learn_vocab",)] = (df, stems, len(problem.documents))
        self.df, self.stems, self.n = cached
        speller = problem.index_cache.get(("speller",))
        if speller is None:
            speller = problem.index_cache[("speller",)] = SpellCorrector(problem.documents)
        self.speller = speller

    def is_gap(self, token: str) -> bool:
        """A word the corpus never uses — and not a misspelling of one: typos are
        spell correction's job; a rule keyed on a typo would never recur."""
        return (len(token) >= 3 and not token.isdigit() and token not in STOPWORDS
                and token not in self.df and light_stem(token) not in self.stems
                and self.speller.correct_token(token) == token)

    def idf(self, token: str) -> float:
        return math.log((self.n + 1) / (self.df.get(token, 0) + 1)) + 1.0


def _target_candidates(problem: SearchProblem, query: Query, vocab: _Vocab) -> list[str]:
    """Corpus words shared by the query's relevant documents, weighted toward the
    most relevant ones and toward distinctive (high-idf) words, so a rule captures
    what the relevant set has in common rather than one document's model number."""
    by_id = {d.doc_id: d for d in problem.documents}
    q_tokens = set(tokenize(query.text))
    weights: Counter[str] = Counter()
    for doc_id, grade in query.qrels.items():
        if grade <= 0:
            continue
        doc = by_id[doc_id]
        for tok in set(split_identifiers(f"{doc.title} {doc.text}")):
            # df >= 2: a rule must bridge to shared vocabulary, not point at one
            # document (a unique identifier or name is retrieval, not a synonym)
            if (tok not in q_tokens and tok not in STOPWORDS and not tok.isdigit()
                    and len(tok) >= 2 and vocab.df.get(tok, 0) >= 2):
                weights[tok] += grade * vocab.idf(tok)
    return [t for t, _ in weights.most_common(_MAX_TARGET_CANDIDATES)]


def learn_rewrites(
    problem: SearchProblem,
    config: PipelineConfig,
    rules: RewriteRules | None = None,
    split: str = "train",
) -> LearningTrace:
    if split != "train":
        raise ValueError("rewrite memory is written from the train split only (plan.md C4)")
    metric = METRICS[problem.primary_metric]
    config = replace(config, use_learned_rewrites=True)
    vocab = _Vocab(problem)
    trace = LearningTrace(rules=rules or RewriteRules())
    history: list[Query] = []

    def score(q: Query, candidate_rules: RewriteRules) -> float:
        pipeline = SearchPipeline(problem.documents, config, candidate_rules, cache=problem.index_cache)
        return metric(pipeline.search(q.text, top_k=100).doc_ids, q.qrels)

    harm: Counter[str] = Counter()
    for q in problem.split(split):
        current = score(q, trace.rules)
        trace.online_scores.append(current)
        history.append(q)

        # Probation: re-verify every rule that fires on this query.
        for tok in dict.fromkeys(tokenize(q.text)):
            rule = trace.rules.get(tok)
            if rule is None or rule.evidence == q.text:
                continue
            without = RewriteRules([r for r in trace.rules if r is not rule])
            delta = current - score(q, without)
            if delta > 1e-9:
                trace.rules = trace.rules.with_rule(replace(rule, support=rule.support + 1))
            elif delta < -1e-9:
                # Refine before demoting: drop one target at a time, and keep the
                # narrower rule if it no longer hurts here.
                narrower = [replace(rule, targets=tuple(t for t in rule.targets if t != drop))
                            for drop in rule.targets] if len(rule.targets) > 1 else []
                fixed = next((r for r in narrower
                              if score(q, trace.rules.with_rule(r)) >= current - delta - 1e-9), None)
                if fixed is not None:
                    trace.rules = trace.rules.with_rule(fixed)
                    current = score(q, trace.rules)
                    continue
                harm[rule.source] += 1
                if harm[rule.source] >= rule.support:
                    trace.rules = without
                    trace.demoted.append(rule)
                    current = score(q, trace.rules)
        if current >= 1.0 - 1e-9:
            continue
        # price grammar ("under $80") is the constraint parser's job, not a synonym
        free_text, _ = parse_constraints(q.text, {})
        sources = [t for t in dict.fromkeys(tokenize(free_text)) if vocab.is_gap(t) and trace.rules.get(t) is None]
        if not sources:
            continue
        targets = _target_candidates(problem, q, vocab)
        best: tuple[float, RewriteRule] | None = None
        for source in sources[:_MAX_SOURCES]:
            chosen: list[str] = []
            best_here = current
            for _ in range(_MAX_TARGETS):
                step = None
                for t in targets:
                    if t in chosen:
                        continue
                    trial = trace.rules.with_rule(RewriteRule(source, tuple(chosen + [t]), 1, 0.0, q.text))
                    s = score(q, trial)
                    if s > best_here + 1e-9:
                        best_here, step = s, t
                if step is None:
                    break
                chosen.append(step)
            if chosen and (best is None or best_here - current > best[0]):
                best = (best_here - current, RewriteRule(source, tuple(chosen), 1, best_here - current, q.text))
        if best is None:
            continue

        # Promotion gate: replay every earlier training query containing the word.
        gain, rule = best
        candidate_rules = trace.rules.with_rule(rule)
        replay = [h for h in history[:-1] if any(light_stem(t) == light_stem(rule.source) for t in tokenize(h.text))]
        deltas = [score(h, candidate_rules) - score(h, trace.rules) for h in replay]
        if any(d < -1e-9 for d in deltas):
            trace.rejected += 1
            continue
        support = 1 + sum(d > 1e-9 for d in deltas)
        rule = replace(rule, support=support, gain=(gain + sum(deltas)) / (1 + len(deltas)))
        trace.rules = trace.rules.with_rule(rule)
        trace.promoted.append(rule)
    return trace
