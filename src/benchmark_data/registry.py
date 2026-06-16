"""Registry of benchmark datasets, normalized to one record schema.

Normalized record:
    {
      "dataset": str,
      "task": "reference_freetext" | "multiple_choice" | "numeric"
              | "hallucination_detection" | "toxicity_generation",
      "question": str,           # base prompt (choices added at format time)
      "choices": list[str]|None,
      "answer": gold (index / number / bool / None),
      "category": str,
      "meta": dict,
    }

Loaders that hit HuggingFace require network access; only the pure normalizers
are exercised by the test suite.
"""
from typing import Dict, Any, List, Callable, Optional

from . import truthfulqa, mmlu, gsm8k, bbq, halueval, toxicity


def _truthfulqa_normalized(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    for r in truthfulqa.load_truthfulqa(limit=limit):
        out.append({
            "dataset": "truthfulqa",
            "task": "reference_freetext",
            "question": r["question"],
            "choices": None,
            "answer": None,
            "category": r.get("category", "general"),
            "meta": {
                "correct_answers": r.get("correct_answers", []),
                "incorrect_answers": r.get("incorrect_answers", []),
                "best_answer": r.get("best_answer"),
            },
        })
    return out


LOADERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "truthfulqa": _truthfulqa_normalized,
    "halueval": halueval.load,
    "mmlu": mmlu.load,
    "gsm8k": gsm8k.load,
    "bbq": bbq.load,
    "toxicity": toxicity.load,
}

# What metric each benchmark is scored by (for docs / dispatch).
TASKS: Dict[str, str] = {
    "truthfulqa": "reference_freetext",
    "halueval": "hallucination_detection",
    "mmlu": "multiple_choice",
    "gsm8k": "numeric",
    "bbq": "multiple_choice",
    "toxicity": "toxicity_generation",
}


def available() -> List[str]:
    return sorted(LOADERS)


def load_benchmark(name: str, limit: Optional[int] = None,
                   split: Optional[str] = None) -> List[Dict[str, Any]]:
    if name not in LOADERS:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available()}")
    loader = LOADERS[name]
    if split is not None:
        try:
            return loader(limit=limit, split=split)
        except TypeError:
            pass
    return loader(limit=limit)


__all__ = ["LOADERS", "TASKS", "available", "load_benchmark"]
