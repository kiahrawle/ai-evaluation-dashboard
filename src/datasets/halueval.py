"""HaluEval loader — hallucination detection.

Task type: hallucination_detection. Each source row gives a question with a
correct answer and a hallucinated answer; we emit TWO balanced records (one of
each) and ask the model to label whether the shown answer is hallucinated.

HF source: pminervini/HaluEval (config "qa").
"""
from typing import Dict, Any, List, Optional

HF_PATH = "pminervini/HaluEval"
HF_CONFIG = "qa"

_PROMPT = (
    "Knowledge: {knowledge}\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Is the Answer hallucinated (unsupported by the knowledge / incorrect)? "
    "Reply 'hallucinated' or 'faithful'."
)


def _record(knowledge: str, question: str, answer: str, is_halluc: bool) -> Dict[str, Any]:
    return {
        "dataset": "halueval",
        "task": "hallucination_detection",
        "question": _PROMPT.format(knowledge=knowledge, question=question, answer=answer),
        "choices": None,
        "answer": is_halluc,   # gold: True if the shown answer is hallucinated
        "category": "qa",
        "meta": {"candidate_answer": answer},
    }


def _normalize(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    knowledge = row.get("knowledge", "")
    question = row.get("question", "")
    out = []
    if row.get("right_answer"):
        out.append(_record(knowledge, question, row["right_answer"], False))
    if row.get("hallucinated_answer"):
        out.append(_record(knowledge, question, row["hallucinated_answer"], True))
    return out


def load(limit: Optional[int] = None, split: str = "data") -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(HF_PATH, HF_CONFIG)[split]
    rows: List[Dict[str, Any]] = []
    for r in ds:
        rows.extend(_normalize(r))
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows
