"""MMLU loader — 4-way multiple-choice across 57 subjects.

Task type: multiple_choice (scored by exact-match accuracy).
HF source: cais/mmlu (config "all").
"""
from typing import Dict, Any, List, Optional

from .cache import cached_or_download

HF_PATH = "cais/mmlu"
HF_CONFIG = "all"
SPLIT = "test"


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    choices: List[str] = list(row.get("choices") or [])
    return {
        "dataset": "mmlu",
        "task": "multiple_choice",
        "question": row.get("question", ""),
        "choices": choices,
        "answer": int(row["answer"]) if row.get("answer") is not None else None,
        "category": row.get("subject", "general"),
        "meta": {},
    }


def _download(limit: Optional[int] = None, split: str = SPLIT) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(HF_PATH, HF_CONFIG)[split]
    rows = [_normalize(r) for r in ds]
    return rows[:limit] if limit else rows


def load(limit: Optional[int] = None, split: str = SPLIT) -> List[Dict[str, Any]]:
    return cached_or_download("mmlu", split, lambda: _download(limit, split), limit)
