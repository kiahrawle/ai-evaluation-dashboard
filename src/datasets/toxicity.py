"""Toxicity loader — prompt continuations scored for toxicity.

Task type: toxicity_generation. The model continues each prompt; the continuation
is scored by a toxicity backend (see src/evaluators/tasks.score_toxicity). The
built-in lexical scorer is a weak placeholder — supply a real classifier
(detoxify / Perspective API) for meaningful numbers.

HF source: allenai/real-toxicity-prompts.
"""
from typing import Dict, Any, List, Optional

HF_PATH = "allenai/real-toxicity-prompts"
HF_CONFIG = None


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = row.get("prompt") or {}
    text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
    seed_tox = prompt.get("toxicity") if isinstance(prompt, dict) else None
    return {
        "dataset": "toxicity",
        "task": "toxicity_generation",
        "question": text,
        "choices": None,
        "answer": None,
        "category": "toxicity",
        "meta": {"prompt_toxicity": seed_tox, "challenging": row.get("challenging")},
    }


def load(limit: Optional[int] = None, split: str = "train") -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(HF_PATH)[split]
    rows = [_normalize(r) for r in ds]
    return rows[:limit] if limit else rows
