"""Leaderboard system for model benchmarking and public rankings."""
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

LEADERBOARD_PATH = Path(__file__).parent.parent.parent / "leaderboard.csv"


def add_result(
    model_name: str,
    hallucination_rate: float,
    groundedness: float,
    avg_risk_score: float,
    citation_coverage: float,
    num_evaluations: int,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Add a model result to the leaderboard.
    
    Returns: result entry with timestamp and rank
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "hallucination_rate": float(hallucination_rate),
        "groundedness": float(groundedness),
        "avg_risk_score": float(avg_risk_score),
        "citation_coverage": float(citation_coverage),
        "num_evaluations": int(num_evaluations),
        "metadata": json.dumps(metadata or {}),
    }
    
    # Append to CSV
    if not LEADERBOARD_PATH.exists():
        with open(LEADERBOARD_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            writer.writeheader()
            writer.writerow(entry)
    else:
        with open(LEADERBOARD_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            writer.writerow(entry)
    
    return entry


def get_leaderboard(limit: int = 50) -> List[Dict[str, Any]]:
    """Get top models ranked by hallucination rate (lowest = best)."""
    if not LEADERBOARD_PATH.exists():
        return []
    
    with open(LEADERBOARD_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Convert numeric fields
    for row in rows:
        row["hallucination_rate"] = float(row.get("hallucination_rate", 1.0))
        row["groundedness"] = float(row.get("groundedness", 0.0))
        row["avg_risk_score"] = float(row.get("avg_risk_score", 1.0))
        row["citation_coverage"] = float(row.get("citation_coverage", 0.0))
        row["num_evaluations"] = int(row.get("num_evaluations", 0))
    
    # Sort by hallucination rate (ascending = better)
    sorted_rows = sorted(rows, key=lambda x: x["hallucination_rate"])
    
    # Add rank
    for i, row in enumerate(sorted_rows[:limit]):
        row["rank"] = i + 1
    
    return sorted_rows[:limit]


def get_model_stats(model_name: str) -> Dict[str, Any]:
    """Get aggregated stats for a specific model across all evaluations."""
    if not LEADERBOARD_PATH.exists():
        return {}
    
    with open(LEADERBOARD_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("model") == model_name]
    
    if not rows:
        return {}
    
    # Calculate averages
    halluc_avg = sum(float(r.get("hallucination_rate", 0)) for r in rows) / len(rows)
    ground_avg = sum(float(r.get("groundedness", 0)) for r in rows) / len(rows)
    risk_avg = sum(float(r.get("avg_risk_score", 0)) for r in rows) / len(rows)
    cite_avg = sum(float(r.get("citation_coverage", 0)) for r in rows) / len(rows)
    total_evals = sum(int(r.get("num_evaluations", 0)) for r in rows)
    
    return {
        "model": model_name,
        "entries": len(rows),
        "total_evaluations": total_evals,
        "avg_hallucination_rate": halluc_avg,
        "avg_groundedness": ground_avg,
        "avg_risk_score": risk_avg,
        "avg_citation_coverage": cite_avg,
        "first_entry": rows[0].get("timestamp"),
        "latest_entry": rows[-1].get("timestamp"),
    }


def compute_leaderboard_score(
    hallucination_rate: float,
    groundedness: float,
    avg_risk_score: float,
    citation_coverage: float,
) -> float:
    """Compute a single composite leaderboard score (0-100 scale).
    
    Weighted formula:
    - Lower hallucination is better (40% weight)
    - Higher groundedness is better (30% weight)
    - Lower risk is better (20% weight)
    - Higher citations is better (10% weight)
    """
    # Normalize scores to 0-100 range
    halluc_normalized = (1.0 - hallucination_rate) * 100
    ground_normalized = groundedness * 100
    risk_normalized = (1.0 - avg_risk_score) * 100
    cite_normalized = citation_coverage * 100
    
    score = (
        halluc_normalized * 0.4 +
        ground_normalized * 0.3 +
        risk_normalized * 0.2 +
        cite_normalized * 0.1
    )
    
    return float(max(0.0, min(100.0, score)))


def get_leaderboard_with_scores(limit: int = 50) -> List[Dict[str, Any]]:
    """Get leaderboard with computed composite scores."""
    leaderboard = get_leaderboard(limit)
    
    for entry in leaderboard:
        entry["leaderboard_score"] = compute_leaderboard_score(
            entry["hallucination_rate"],
            entry["groundedness"],
            entry["avg_risk_score"],
            entry["citation_coverage"],
        )
    
    # Re-sort by score
    leaderboard = sorted(leaderboard, key=lambda x: x["leaderboard_score"], reverse=True)
    
    # Update ranks
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1
    
    return leaderboard


__all__ = [
    "add_result",
    "get_leaderboard",
    "get_model_stats",
    "compute_leaderboard_score",
    "get_leaderboard_with_scores",
]
