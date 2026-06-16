"""Validate the local RAG-groundedness and risk engines against human labels.

The README's design philosophy says: don't trust a roadmap layer until it's
checked against ground truth. This is that check for the groundedness and risk
engines (the judge has its own check in judge_validation.py).

Input CSV (see data/validation_human_labels.csv):

    question, response, context, factual_groundedness, risk_level

- `context` is the supporting evidence the groundedness engine retrieves against.
- `factual_groundedness` is 1/0 (human: is the response supported?).
- `risk_level` is LOW/MEDIUM/HIGH (human: hallucination risk).

It runs the real engines (`evaluators.verify_citation_support` /
`score_groundedness`, `risk.score_risk_with_details` fed by local signals) and
reports:

- a confusion matrix for groundedness (binary) and risk band (3-class), and
- the Pearson correlation between the engine's continuous risk score and the
  ordinal human risk level (LOW=0, MEDIUM=1, HIGH=2).
"""
from typing import Dict, Any, List, Sequence
from pathlib import Path
import math
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src import evaluators, risk
from src import utils

RISK_ORDINAL = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_ORDINAL_RISK = {v: k for k, v in RISK_ORDINAL.items()}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if either side has no
    variance (correlation undefined)."""
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def confusion_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                     labels: Sequence[str]) -> Dict[str, Any]:
    idx = {l: i for i, l in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    correct = sum(mat[i][i] for i in range(len(labels)))
    total = sum(sum(r) for r in mat)
    return {"labels": list(labels), "matrix": mat,
            "accuracy": correct / total if total else 0.0, "n": total}


def _risk_band(score: float) -> str:
    return "HIGH" if score >= 0.67 else "MEDIUM" if score >= 0.33 else "LOW"


def predict_row(response: str, context: str) -> Dict[str, Any]:
    """Run the local engines on one (response, context) pair."""
    docs = [{"text": context}] if context and str(context).strip() else []
    cite = evaluators.verify_citation_support(response, docs)
    grounded_score = evaluators.score_groundedness(docs, response) if docs else 0.0
    markers = utils.scan_for_risk_markers(response)
    contradiction = markers["contradictions"]["contradiction_score"]
    uncertainty = markers["uncertainty"]["uncertainty_score"]
    support_score = cite.get("support_score") or 0.0

    details = risk.score_risk_with_details(
        hallucination_score=1.0 - grounded_score,   # ungrounded => likely hallucination
        grounded_score=grounded_score,
        confidence_score=max(0.0, 1.0 - uncertainty),
        unsupported_claims_ratio=1.0 - support_score,
        contradiction_score=contradiction,
    )
    return {
        "pred_grounded": 1 if cite.get("citation_supported") else 0,
        "risk_score": details["risk_score"],
        "pred_risk_level": details["risk_level"],
    }


def validate_framework(labeled_csv: str | Path) -> Dict[str, Any]:
    df = pd.read_csv(labeled_csv)
    required = {"response", "context", "factual_groundedness", "risk_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    g_true, g_pred = [], []
    r_true, r_pred = [], []
    risk_scores, risk_true_ord = [], []

    for _, row in df.iterrows():
        pred = predict_row(str(row["response"]), str(row.get("context", "")))
        g_true.append(int(row["factual_groundedness"]))
        g_pred.append(pred["pred_grounded"])
        human_band = str(row["risk_level"]).strip().upper()
        r_true.append(human_band)
        r_pred.append(pred["pred_risk_level"])
        risk_scores.append(pred["risk_score"])
        risk_true_ord.append(RISK_ORDINAL.get(human_band, 1))

    ground_cm = confusion_binary(g_true, g_pred)
    risk_cm = confusion_matrix(r_true, r_pred, ["LOW", "MEDIUM", "HIGH"])
    r = pearson(risk_scores, risk_true_ord)

    _print_report(ground_cm, risk_cm, r, len(df))
    return {"n": len(df), "groundedness": ground_cm, "risk": risk_cm,
            "risk_pearson": r}


def _print_report(ground_cm, risk_cm, pearson_r, n):
    print(f"\n=== Framework validation ({n} labelled rows) ===")
    print("\n-- Groundedness engine vs human factual_groundedness --")
    print(f"  accuracy {ground_cm['accuracy']:.2f}  precision {ground_cm['precision']:.2f}"
          f"  recall {ground_cm['recall']:.2f}  f1 {ground_cm['f1']:.2f}")
    print(f"  TP {ground_cm['tp']}  TN {ground_cm['tn']}  "
          f"FP {ground_cm['fp']}  FN {ground_cm['fn']}")

    print("\n-- Risk engine band vs human risk_level (rows=human, cols=pred) --")
    labels = risk_cm["labels"]
    print("            " + "".join(f"{l:>8}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"  {lab:>8}  " + "".join(f"{risk_cm['matrix'][i][j]:>8}" for j in range(len(labels))))
    print(f"  band accuracy: {risk_cm['accuracy']:.2f}")
    print(f"\n  Pearson(risk_score, human risk ordinal): {pearson_r:.3f}")
    if pearson_r < 0.3:
        print("  NOTE: weak correlation — the risk engine does not track human risk "
              "judgments well on this set. Inspect before relying on risk scores.")


__all__ = ["pearson", "confusion_binary", "confusion_matrix",
           "predict_row", "validate_framework", "RISK_ORDINAL"]
