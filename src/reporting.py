"""Pure helpers that turn stored EvaluationResult dicts into dashboard-ready
numbers. Kept free of Streamlit so they can be unit-tested directly.

The dashboard reads `results/results.json` (written by analyze.summarize). When
it is absent every helper returns an empty/zero result so the UI can show an
honest "no data yet" state instead of fabricated metrics.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import ast
import json

import config


def load_results(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load stored evaluation results.

    Prefers the structured `results.json`; if that is absent, falls back to
    `raw_results.csv` so a dashboard still shows data from older runs. Returns
    [] when no results exist yet.
    """
    p = Path(path) if path else (config.RESULTS_DIR / "results.json")
    if p.exists():
        try:
            with p.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    # Fallback: reconstruct minimal records from the flat CSV.
    csv_path = (p.parent / "raw_results.csv") if path else (config.RESULTS_DIR / "raw_results.csv")
    if csv_path.exists():
        return _load_from_csv(csv_path)
    return []


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1"}:
        return True
    if s in {"false", "0"}:
        return False
    return None


def _load_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        meta = {}
        raw_meta = row.get("metadata")
        if isinstance(raw_meta, str) and raw_meta.strip().startswith("{"):
            try:
                meta = ast.literal_eval(raw_meta)
            except Exception:
                meta = {}
        if "category" not in meta and "category" in df.columns:
            meta["category"] = row.get("category")
        records.append({
            "truthful": _to_bool(row.get("truthful")),
            "grounded": _to_bool(row.get("grounded")),
            "risk_score": row.get("risk_score"),
            "model": row.get("model"),
            "metadata": meta,
        })
    return records


def _meta(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metadata")
    return m if isinstance(m, dict) else {}


def _mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def overview_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline numbers for the dashboard, computed from real results."""
    n = len(results)
    if n == 0:
        return {
            "total": 0, "hallucination_rate": 0.0, "avg_risk_score": 0.0,
            "citation_coverage": 0.0, "dangerous_high": 0, "contradictions": 0,
        }
    hallucinated = sum(1 for r in results if r.get("truthful") is False)
    citation_flags = [_meta(r).get("citation_supported") for r in results]
    citation_flags = [c for c in citation_flags if c is not None]
    return {
        "total": n,
        "hallucination_rate": hallucinated / n,
        "avg_risk_score": _mean([r.get("risk_score") for r in results]),
        "citation_coverage": (
            sum(1 for c in citation_flags if c) / len(citation_flags)
            if citation_flags else 0.0
        ),
        "dangerous_high": sum(1 for r in results if _meta(r).get("dangerous_level") == "HIGH"),
        "contradictions": sum(1 for r in results if _meta(r).get("numeric_contradiction")),
    }


def risk_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count of LOW/MEDIUM/HIGH risk levels (from metadata.risk_level)."""
    dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in results:
        level = _meta(r).get("risk_level")
        if level in dist:
            dist[level] += 1
    return dist


def hallucination_by_category(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean hallucination rate per question category."""
    buckets: Dict[str, List[int]] = {}
    for r in results:
        cat = _meta(r).get("category") or "unknown"
        buckets.setdefault(cat, []).append(1 if r.get("truthful") is False else 0)
    return {cat: sum(v) / len(v) for cat, v in buckets.items() if v}


def leaderboard_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-model leaderboard rows from stored results.

    Ranked by hallucination rate (lower is better).
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_model.setdefault(r.get("model") or "unknown", []).append(r)

    rows = []
    for model, rs in by_model.items():
        n = len(rs)
        hallucinated = sum(1 for r in rs if r.get("truthful") is False)
        grounded = sum(1 for r in rs if r.get("grounded"))
        cites = [_meta(r).get("citation_supported") for r in rs]
        cites = [c for c in cites if c is not None]
        rows.append({
            "model": model,
            "hallucination_rate": hallucinated / n if n else 0.0,
            "groundedness": grounded / n if n else 0.0,
            "avg_risk_score": _mean([r.get("risk_score") for r in rs]),
            "citation_coverage": (sum(1 for c in cites if c) / len(cites)) if cites else 0.0,
            "evaluations": n,
        })
    rows.sort(key=lambda x: x["hallucination_rate"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


__all__ = [
    "load_results", "overview_metrics", "risk_distribution",
    "hallucination_by_category", "leaderboard_from_results",
]
