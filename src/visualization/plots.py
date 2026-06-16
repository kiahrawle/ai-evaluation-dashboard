"""Simple plotting helpers for results CSVs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_hallucination_by_category(results_csv: str | Path, out_path: str | Path = None):
    df = pd.read_csv(results_csv)
    if "category" not in df.columns or "hallucinated" not in df.columns:
        raise ValueError("Expected CSV with 'category' and 'hallucinated' columns")
    pivot = (df.groupby(["category", "model"])["hallucinated"].mean().unstack("model").sort_index())
    height = max(4, 0.45 * len(pivot))
    ax = pivot.plot(kind="barh", figsize=(9, height))
    ax.set_xlabel("Hallucination rate (lower is better)")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("Hallucination rate by topic category")
    ax.legend(title="model")
    plt.tight_layout()
    out = out_path or (Path(results_csv).parent / "hallucination_by_category.png")
    plt.savefig(out, dpi=150)
    return out


__all__ = ["plot_hallucination_by_category"]
