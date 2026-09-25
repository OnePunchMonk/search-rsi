"""The search use-case suite.

Each built-in problem is a deterministic generator for one family of search
behavior (see docs/USE_CASES.md); `load_problem` also accepts a path to any
BEIR-format directory, so the same optimizer and memory machinery runs on real
datasets without code changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from search_rsi.problems import code, entity, faq, multihop, news, product
from search_rsi.problems.base import SPLIT_NAMES, Query, SearchProblem, hash_splits, round_robin_splits
from search_rsi.problems.beir import load_beir_dir

BUILTIN_PROBLEMS: dict[str, Callable[[], SearchProblem]] = {
    "product_search": product.build,
    "code_search": code.build,
    "faq_support": faq.build,
    "entity_lookup": entity.build,
    "news_freshness": news.build,
    "multihop_provenance": multihop.build,
}

_cache: dict[str, SearchProblem] = {}


def list_problems() -> list[str]:
    return list(BUILTIN_PROBLEMS)


def load_problem(name_or_path: str) -> SearchProblem:
    """A built-in problem by name, or a BEIR-format directory by path. Built
    problems are cached per process so their index caches are shared."""
    if name_or_path not in _cache:
        if name_or_path in BUILTIN_PROBLEMS:
            _cache[name_or_path] = BUILTIN_PROBLEMS[name_or_path]()
        elif Path(name_or_path).is_dir():
            _cache[name_or_path] = load_beir_dir(name_or_path)
        else:
            raise ValueError(f"unknown problem {name_or_path!r}: not one of {list_problems()} and not a directory")
    return _cache[name_or_path]


__all__ = [
    "BUILTIN_PROBLEMS",
    "SPLIT_NAMES",
    "Query",
    "SearchProblem",
    "hash_splits",
    "list_problems",
    "load_beir_dir",
    "load_problem",
    "round_robin_splits",
]
