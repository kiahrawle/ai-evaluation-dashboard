"""Dataset loaders for the platform.

TruthfulQA is the validated core; HaluEval / MMLU / GSM8K / BBQ / toxicity are
additional benchmarks normalized to one record schema via `registry`, with
task-appropriate scoring in `src.evaluators.tasks` and a generic `runner`.
"""
from .truthfulqa import load_truthfulqa, coarse_category
from .registry import available, load_benchmark, TASKS
from .runner import run_benchmark, run_records

__all__ = [
    "load_truthfulqa", "coarse_category",
    "available", "load_benchmark", "TASKS",
    "run_benchmark", "run_records",
]
