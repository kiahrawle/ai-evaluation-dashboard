"""Validate the LLM judge against human labels before trusting its numbers.

The README is blunt about this: "Before reporting any numbers, hand-label ~50
answers yourself and check the judge's agreement with your labels (Cohen's
kappa). This step is what separates a real eval from a vibe." This module is
that step.

Input CSV (one row per hand-labelled answer):

    question, answer, human_truthful, correct_answers, incorrect_answers

`human_truthful` is 1/0 (or true/false). `correct_answers` / `incorrect_answers`
are optional "||"-separated reference lists the judge needs to make its call.
"""
from typing import Dict, Any, List, Sequence
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src import scoring


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa for two raters over matched, categorical labels.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative below
    chance. When both raters use a single category (pe == 1) kappa is undefined;
    we return 1.0 if they fully agree, else 0.0.
    """
    a = list(a)
    b = list(b)
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("rater label lists must be non-empty and equal length")

    po = sum(1 for x, y in zip(a, b) if x == y) / n

    categories = set(a) | set(b)
    pe = 0.0
    for c in categories:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for y in b if y == c) / n
        pe += pa * pb

    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _split_refs(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [s.strip() for s in str(value).split("||") if s.strip()]


def validate_judge(labeled_csv: str | Path, judge_model: str | None = None) -> Dict[str, Any]:
    """Run the judge over a hand-labelled CSV and report agreement + kappa."""
    if judge_model:
        scoring.config.JUDGE_MODEL = judge_model

    df = pd.read_csv(labeled_csv)
    required = {"question", "answer", "human_truthful"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    cache: Dict[str, Any] = {}
    human: List[bool] = []
    judge: List[bool] = []
    rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        item = {
            "question": r["question"],
            "correct_answers": _split_refs(r.get("correct_answers")),
            "incorrect_answers": _split_refs(r.get("incorrect_answers")),
        }
        verdict = scoring.judge_answer(item, str(r["answer"]), cache)
        h = _to_bool(r["human_truthful"])
        j = bool(verdict.get("truthful", True))
        human.append(h)
        judge.append(j)
        rows.append({"question": r["question"], "human_truthful": h, "judge_truthful": j})

    n = len(human)
    agreement = sum(1 for h, j in zip(human, judge) if h == j) / n if n else 0.0
    kappa = cohens_kappa(human, judge) if n else 0.0

    interp = (
        "poor" if kappa < 0.2 else
        "fair" if kappa < 0.4 else
        "moderate" if kappa < 0.6 else
        "substantial" if kappa < 0.8 else
        "almost perfect"
    )

    print("\n=== Judge validation ===")
    print(f"judge model:        {judge_model or scoring.config.JUDGE_MODEL}")
    print(f"labelled answers:   {n}")
    print(f"raw agreement:      {agreement:.1%}")
    print(f"Cohen's kappa:      {kappa:.3f}  ({interp})")
    if kappa < 0.4:
        print("WARNING: agreement is weak. Fix the judge prompt before trusting "
              "any hallucination numbers it produces.")

    return {"n": n, "agreement": agreement, "cohens_kappa": kappa,
            "interpretation": interp, "rows": rows}


__all__ = ["cohens_kappa", "validate_judge"]
