"""Aggregate results: per-category + broad-bucket hallucination rates, with
Wilson 95% confidence intervals so small samples don't mislead you."""
import sys
import math
import json
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config
from src.benchmark_data import coarse_category


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (k successes out of n).
    Better than +/- sqrt(p(1-p)/n) when n is small or p is near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize(all_results: list[dict]) -> pd.DataFrame:
    # `all_results` is a list of EvaluationResult dicts (the unified schema).
    # The judge verdict lives in `truthful`/`informative`; the cheap semantic
    # baseline verdict is carried in `grounded`; `category` is nested in
    # `metadata` (it is not a question-level field on the schema).
    df = pd.DataFrame(all_results)
    if "category" not in df.columns:
        df["category"] = df["metadata"].map(
            lambda m: (m or {}).get("category") if isinstance(m, dict) else None
        )
    truthful = df["truthful"].astype(bool)
    informative = df["informative"].astype(bool)
    semantic = df["grounded"].astype(bool)
    df["hallucinated"] = ~truthful
    df["truthful_and_informative"] = truthful & informative
    df["judge_vs_semantic_agree"] = truthful == semantic
    df["bucket"] = df["category"].map(coarse_category)

    # ---- overall, WITH confidence intervals ----
    rows = []
    for model, g in df.groupby("model"):
        n = len(g)
        k = int(g["hallucinated"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            "model": model,
            "n": n,
            "hallucination_rate": round(k / n, 3),
            "halluc_95ci": f"[{lo:.2f}, {hi:.2f}]",
            "truthful_informative_rate": round(g["truthful_and_informative"].mean(), 3),
            "judge_semantic_agreement": round(g["judge_vs_semantic_agree"].mean(), 3),
        })
    overall = pd.DataFrame(rows).set_index("model")

    by_cat = (df.groupby(["category", "model"])["hallucinated"]
              .mean().unstack("model").round(3).sort_index())
    by_bucket = (df.groupby(["bucket", "model"])["hallucinated"]
                 .mean().unstack("model").round(3))

    print("\n=== Overall (with 95% confidence interval) ===")
    print(overall.to_string())
    print("\n=== Hallucination rate by BROAD bucket ===")
    print(by_bucket.to_string())
    print("\n=== Hallucination rate by category ===")
    print(by_cat.to_string())

    df.to_csv(config.RESULTS_DIR / "raw_results.csv", index=False)
    by_cat.to_csv(config.RESULTS_DIR / "comparison.csv")
    by_bucket.to_csv(config.RESULTS_DIR / "buckets.csv")
    overall.to_csv(config.RESULTS_DIR / "overall.csv")
    # Structured dump that preserves nested metadata (risk_level, dangerous_level,
    # numeric_contradiction, citation_evidence, ...) for the dashboard, which a
    # flat CSV would stringify and lose.
    with (config.RESULTS_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nSaved CSVs + results.json to {config.RESULTS_DIR}/")
    return df
