"""BBQ loader — Bias Benchmark for QA (social-bias multiple choice).

Task type: multiple_choice (accuracy here; a dedicated bias score is roadmap).
HF source: the canonical fields are context / question / ans0..ans2 / label /
category / context_condition. The HF path varies by mirror, so it is a constant
you may need to adjust for your environment.
"""
from typing import Dict, Any, List, Optional

from .cache import cached_or_download

# Adjust if your mirror differs (e.g. "Elfsong/BBQ", "walledai/BBQ").
HF_PATH = "heegyu/bbq"
HF_CONFIG = None
SPLIT = "test"


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = [row.get("ans0", ""), row.get("ans1", ""), row.get("ans2", "")]
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    prompt = f"{context}\n{question}".strip() if context else question
    label = row.get("label")
    return {
        "dataset": "bbq",
        "task": "multiple_choice",
        "question": prompt,
        "choices": choices,
        "answer": int(label) if label is not None else None,
        "category": row.get("category", "bias"),
        "meta": {
            "context_condition": row.get("context_condition"),
            "question_polarity": row.get("question_polarity"),
        },
    }


def _download(limit: Optional[int] = None, split: str = SPLIT) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(HF_PATH, HF_CONFIG)[split] if HF_CONFIG else load_dataset(HF_PATH)[split]
    rows = [_normalize(r) for r in ds]
    return rows[:limit] if limit else rows


def load(limit: Optional[int] = None, split: str = SPLIT) -> List[Dict[str, Any]]:
    return cached_or_download("bbq", split, lambda: _download(limit, split), limit)
