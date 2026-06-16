"""GSM8K loader — grade-school math word problems.

Task type: numeric (extract final number, exact match).
HF source: gsm8k (config "main"). Gold answers carry a "#### <number>" marker.
"""
from typing import Dict, Any, List, Optional
import re

from .cache import cached_or_download

HF_PATH = "gsm8k"
HF_CONFIG = "main"
SPLIT = "test"


def _gold_number(answer_field: str) -> Optional[str]:
    if answer_field and "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    return answer_field


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": "gsm8k",
        "task": "numeric",
        "question": row.get("question", ""),
        "choices": None,
        "answer": _gold_number(row.get("answer", "")),
        "category": "math",
        "meta": {"solution": row.get("answer", "")},
    }


def _download(limit: Optional[int] = None, split: str = SPLIT) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(HF_PATH, HF_CONFIG)[split]
    rows = [_normalize(r) for r in ds]
    return rows[:limit] if limit else rows


def load(limit: Optional[int] = None, split: str = SPLIT) -> List[Dict[str, Any]]:
    return cached_or_download("gsm8k", split, lambda: _download(limit, split), limit)
