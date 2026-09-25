"""Recursive self-improvement for search pipelines, at three levels:

1. `learn` — inner loop: query-rewrite rules learned from training queries.
2. `optimize` — outer loop: evolve the pipeline configuration itself.
3. `transfer` — cross-problem memory: champions warm-start similar problems.
"""
from search_rsi.rsi.evaluate import PipelineScore, evaluate_pipeline
from search_rsi.rsi.export import export_pipeline
from search_rsi.rsi.learn import LearningTrace, learn_rewrites
from search_rsi.rsi.optimize import (
    OptimizeResult,
    optimize_pipeline,
    propose_pipeline_mutation,
    random_pipeline_config,
)
from search_rsi.rsi.transfer import ConfigMemory, ProblemProfile, profile_problem

__all__ = [
    "ConfigMemory",
    "LearningTrace",
    "OptimizeResult",
    "PipelineScore",
    "ProblemProfile",
    "evaluate_pipeline",
    "export_pipeline",
    "learn_rewrites",
    "optimize_pipeline",
    "profile_problem",
    "propose_pipeline_mutation",
    "random_pipeline_config",
]
