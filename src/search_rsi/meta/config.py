from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessConfig:
    """The full mutable surface of the meta-loop, v1 (numeric only).

    Everything the meta-loop is allowed to touch lives here. Anything not listed —
    the grader, the gold task set, the split boundaries, the tool-budget accounting —
    is out of the search space by construction: there is no field for it, so no
    mutation can reach it (docs/META_RSI.md, Guardrail 1).
    """

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    verifier_min_overlap: int = 1

    def label(self) -> str:
        return f"k1={self.bm25_k1:.2f},b={self.bm25_b:.2f},min_overlap={self.verifier_min_overlap}"


BASELINE = HarnessConfig()
