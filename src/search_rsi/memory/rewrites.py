"""Learned query-rewrite rules: the search pipeline's cross-run experience memory.

A rule says "when a user writes `sneakers`, also search for `running shoes`". Rules
are mined from *training* queries the current pipeline failed on, verified by
replaying the training history before they are promoted, and persisted as plain
JSON (plan.md C3: improvement lives in inspectable artifacts, never in weights).
Held-out queries may read rules; nothing on the eval path can write them (C4).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from search_rsi.text import light_stem, tokenize


@dataclass(frozen=True)
class RewriteRule:
    source: str  # user-vocabulary token, as typed
    targets: tuple[str, ...]  # corpus-vocabulary tokens appended when source appears
    support: int  # training queries on which the rule measurably helped
    gain: float  # mean primary-metric gain on those queries
    evidence: str  # the first training query that motivated it


class RewriteRules:
    """An immutable-by-convention rule set, applied at query time."""

    def __init__(self, rules: list[RewriteRule] | None = None):
        self._rules: dict[str, RewriteRule] = {}
        for rule in rules or []:
            self._rules[light_stem(rule.source)] = rule

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules.values())

    def get(self, token: str) -> RewriteRule | None:
        return self._rules.get(light_stem(token))

    def with_rule(self, rule: RewriteRule) -> "RewriteRules":
        return RewriteRules([r for r in self if light_stem(r.source) != light_stem(rule.source)] + [rule])

    def apply(self, query: str) -> str:
        """Expansion, not replacement: the user's own words stay in the query."""
        tokens = tokenize(query)
        present = set(tokens)
        extra: list[str] = []
        for tok in tokens:
            rule = self.get(tok)
            if rule is None:
                continue
            for target in rule.targets:
                if target not in present and target not in extra:
                    extra.append(target)
        return f"{query} {' '.join(extra)}" if extra else query

    def to_json(self) -> list[dict]:
        return [dict(asdict(r), targets=list(r.targets)) for r in sorted(self, key=lambda r: r.source)]

    @classmethod
    def from_json(cls, data: list[dict]) -> "RewriteRules":
        return cls([RewriteRule(**dict(d, targets=tuple(d["targets"]))) for d in data])


class RewriteMemory:
    """Persisted rule sets, one namespace per search problem, in one JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load_all(self) -> dict[str, list[dict]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def read(self, namespace: str) -> RewriteRules:
        return RewriteRules.from_json(self._load_all().get(namespace, []))

    def write(self, namespace: str, rules: RewriteRules) -> None:
        data = self._load_all()
        data[namespace] = rules.to_json()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
