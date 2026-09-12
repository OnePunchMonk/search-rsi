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

    retriever: str = "bm25"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    verifier_min_overlap: int = 1
    use_hyde: bool = False

    def label(self) -> str:
        params = f"min_overlap={self.verifier_min_overlap}"
        if self.retriever == "bm25":
            params = f"k1={self.bm25_k1:.2f},b={self.bm25_b:.2f},{params}"
        hyde = "+hyde" if self.use_hyde else ""
        return f"{self.retriever}{hyde}[{params}]"


BASELINE = HarnessConfig()
