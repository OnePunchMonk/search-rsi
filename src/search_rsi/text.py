"""Text analysis: the tokenizer choices the pipeline optimizer can search over.

Different search use cases want different analyzers — code search needs
`parseConfigFile` split into `parse config file`, product search wants `shoes` and
`shoe` to match, FAQ matching wants stopwords gone. Which analyzer wins on a given
problem is something the optimizer measures (docs/USE_CASES.md), so every analyzer
here is a plain function `str -> list[str]` registered by name.
"""
from __future__ import annotations

import re
from typing import Callable

_WORD_RE = re.compile(r"[a-z0-9]+")
_IDENT_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i in is it me my of on or
    that the this to was what when where which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def split_identifiers(text: str) -> list[str]:
    """Splits camelCase / PascalCase / snake_case identifiers into words, and keeps
    the joined identifier too so an exact-name query still matches."""
    out: list[str] = []
    for chunk in _IDENT_RE.findall(text):
        parts = [p.lower() for p in _CAMEL_RE.findall(chunk)]
        out.extend(parts)
        if len(parts) > 1:
            out.append(chunk.lower())
    return out


def light_stem(token: str) -> str:
    """A tiny suffix stripper (plural / -ing / -ed / trailing e). Not Porter, but
    consistent on both sides of the match, which is all ranking needs."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            if len(token) >= 3 and token[-1] == token[-2] and token[-1] not in "lsz":
                token = token[:-1]
            break
    if token.endswith("e") and len(token) >= 4:
        token = token[:-1]
    return token


def char_ngrams(tokens: list[str], n: int = 3) -> list[str]:
    grams: list[str] = []
    for tok in tokens:
        padded = f"#{tok}#"
        if len(padded) <= n:
            grams.append(padded)
        else:
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


def _plain(text: str) -> list[str]:
    return tokenize(text)


def _stem(text: str) -> list[str]:
    return [light_stem(t) for t in tokenize(text)]


def _stem_stop(text: str) -> list[str]:
    return [light_stem(t) for t in tokenize(text) if t not in STOPWORDS]


def _code(text: str) -> list[str]:
    return [light_stem(t) for t in split_identifiers(text) if t not in STOPWORDS]


def _trigram(text: str) -> list[str]:
    return char_ngrams(tokenize(text))


ANALYZERS: dict[str, Callable[[str], list[str]]] = {
    "plain": _plain,
    "stem": _stem,
    "stem_stop": _stem_stop,
    "code": _code,
}

# Not a word analyzer the optimizer picks directly — the `charngram` retriever uses it.
TRIGRAM_ANALYZER = _trigram


def get_analyzer(name: str) -> Callable[[str], list[str]]:
    try:
        return ANALYZERS[name]
    except KeyError:
        raise ValueError(f"unknown analyzer: {name!r} (known: {sorted(ANALYZERS)})") from None


def edit_distance(a: str, b: str, limit: int = 3) -> int:
    """Damerau-Levenshtein (optimal string alignment), with an early exit once the
    distance provably exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]
