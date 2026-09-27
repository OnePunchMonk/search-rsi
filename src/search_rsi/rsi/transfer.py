"""Cross-problem memory: what worked on which kind of search problem.

The third RSI level. Every optimized problem leaves behind its champion pipeline
and a profile of the problem's shape (document length, identifier density,
structured fields, how often queries use words the corpus never does...). A new
problem is profiled the same way, and the champions of the most similar known
problems become warm starts for its optimizer — or, with zero search budget, the
pipeline it ships with. Profiles read only documents and *training* queries, so
nothing about meta_val or held-out eval leaks into the suggestion.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from search_rsi.pipeline import PipelineConfig
from search_rsi.pipeline.query import SpellCorrector, has_temporal_intent
from search_rsi.problems.base import SearchProblem
from search_rsi.text import STOPWORDS, tokenize

_IDENT = re.compile(r"\b[a-z]+[A-Z][A-Za-z]*\b|\b\w+_\w+\b")


@dataclass(frozen=True)
class ProblemProfile:
    log_doc_len: float
    identifier_density: float
    price_field: float
    date_field: float
    title_field: float
    query_gap_rate: float  # share of train-query words absent from the corpus
    query_typo_rate: float  # ... of which are one or two edits from a corpus word
    query_temporal_rate: float  # share of train queries with a recency cue
    log_query_len: float
    query_digit_rate: float

    def distance(self, other: "ProblemProfile") -> float:
        a, b = asdict(self), asdict(other)
        return sum(abs(a[k] - b[k]) for k in a)


def profile_problem(problem: SearchProblem) -> ProblemProfile:
    docs = problem.documents
    vocab: set[str] = set()
    lengths = []
    idents = 0
    raw_tokens = 0
    for d in docs:
        toks = tokenize(f"{d.title} {d.text}")
        vocab.update(toks)
        lengths.append(len(toks))
        idents += len(_IDENT.findall(f"{d.title} {d.text}"))
        raw_tokens += len(toks)
    train = problem.split("train")
    q_tokens = [t for q in train for t in tokenize(q.text) if t not in STOPWORDS]
    n_train = max(1, len(train))
    gaps = [t for t in q_tokens if t not in vocab]
    speller = problem.index_cache.get(("speller",))
    if speller is None:
        speller = problem.index_cache[("speller",)] = SpellCorrector(docs)
    typos = sum(speller.correct_token(t) != t for t in gaps)

    def share(key: str) -> float:
        return sum(isinstance(d.metadata.get(key), (int, float)) for d in docs) / len(docs)

    return ProblemProfile(
        log_doc_len=math.log1p(sum(lengths) / len(lengths)) / 5,
        identifier_density=min(1.0, 10 * idents / max(1, raw_tokens)),
        price_field=share("price"),
        date_field=share("date"),
        title_field=sum(bool(d.title) for d in docs) / len(docs),
        query_gap_rate=len(gaps) / max(1, len(q_tokens)),
        query_typo_rate=typos / max(1, len(gaps)),
        query_temporal_rate=sum(has_temporal_intent(q.text) for q in train) / n_train,
        log_query_len=math.log1p(len(q_tokens) / n_train) / 3,
        query_digit_rate=sum(t.isdigit() for t in q_tokens) / max(1, len(q_tokens)),
    )


class ConfigMemory:
    """JSON store of {problem, profile, champion config, scores}."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def record(self, problem: SearchProblem, config: PipelineConfig, meta_val: float, held_out: float | None) -> None:
        entries = [e for e in self.entries() if e["problem"] != problem.name]
        entries.append({
            "problem": problem.name,
            "use_case": problem.use_case,
            "profile": asdict(profile_problem(problem)),
            "config": config.to_dict(),
            "meta_val_score": meta_val,
            "held_out_score": held_out,
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2) + "\n")

    def suggest(self, problem: SearchProblem, k: int = 2, exclude_self: bool = True) -> list[tuple[str, PipelineConfig, float]]:
        """Champions of the k most similar recorded problems: (name, config, distance)."""
        target = profile_problem(problem)
        ranked = []
        for e in self.entries():
            if exclude_self and e["problem"] == problem.name:
                continue
            dist = target.distance(ProblemProfile(**e["profile"]))
            ranked.append((dist, e["problem"], PipelineConfig.from_dict(e["config"])))
        ranked.sort(key=lambda r: r[0])
        return [(name, config, dist) for dist, name, config in ranked[:k]]
