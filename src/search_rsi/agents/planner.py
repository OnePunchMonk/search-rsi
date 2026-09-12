from __future__ import annotations

from search_rsi.memory import MemoryEntry
from search_rsi.types import Task


class RuleBasedPlanner:
    """Deterministic stand-in for an LLM planner.

    Exercises the harness's plan -> retrieve -> verify -> answer contract without an
    API call, exactly as the sibling diffusion project's local deterministic backend
    stands in for a real denoiser. Swapping this for an LLM planner should require no
    change to Harness (plan.md, First executable slice / Milestones).
    """

    def plan_queries(self, task: Task, memory: list[MemoryEntry]) -> list[str]:
        queries = [task.question]
        for entry in memory:
            if "decompose" in entry.strategy.lower():
                # naive decomposition: split on conjunctions as a stand-in for
                # real multi-hop query planning
                for part in task.question.replace("?", "").split(" and "):
                    part = part.strip()
                    if part and part not in queries:
                        queries.append(part)
        return queries
