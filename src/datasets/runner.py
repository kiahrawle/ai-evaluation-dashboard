"""Generic benchmark runner: feed normalized records through a generation
function and a task scorer, then aggregate task-appropriate metrics.

`generate_fn` is injectable (`str -> str`), so the suite is testable offline with
a stub and runs live with `models.chat`.
"""
from typing import Callable, List, Dict, Any, Optional

from src.evaluators.tasks import format_prompt, score_record


def aggregate(per_example: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_example)
    task = per_example[0]["task"] if per_example else None
    out: Dict[str, Any] = {"n": n, "task": task}

    graded = [p["correct"] for p in per_example if p.get("correct") is not None]
    if graded:
        out["scored"] = len(graded)
        out["accuracy"] = sum(1 for c in graded if c) / len(graded)

    tox = [p["toxicity"] for p in per_example if "toxicity" in p]
    if tox:
        out["mean_toxicity"] = sum(tox) / len(tox)
        out["toxic_rate"] = sum(1 for p in per_example if p.get("toxic")) / n if n else 0.0

    out["per_example"] = per_example
    return out


def run_records(records: List[Dict[str, Any]],
                generate_fn: Callable[[str], str],
                toxicity_backend: Optional[Callable[[str], float]] = None) -> Dict[str, Any]:
    per_example = []
    for rec in records:
        pred = generate_fn(format_prompt(rec))
        scored = score_record(rec, pred, toxicity_backend=toxicity_backend)
        scored["dataset"] = rec.get("dataset")
        scored["category"] = rec.get("category")
        per_example.append(scored)
    return aggregate(per_example)


def run_benchmark(name: str, generate_fn: Callable[[str], str],
                  limit: Optional[int] = None, split: Optional[str] = None,
                  toxicity_backend: Optional[Callable[[str], float]] = None) -> Dict[str, Any]:
    from .registry import load_benchmark
    records = load_benchmark(name, limit=limit, split=split)
    result = run_records(records, generate_fn, toxicity_backend=toxicity_backend)
    result["dataset"] = name
    return result


__all__ = ["run_records", "run_benchmark", "aggregate"]
