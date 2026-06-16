"""Turn the per-category results into a horizontal bar chart.

    python plot.py

Reads results/raw_results.csv (written by a normal run) and saves
results/hallucination_by_category.png.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")        # render to file, no GUI window needed
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
import config


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:   # CSV stores booleans as the text "True"/"False"
        return series.astype(str).str.strip().str.lower().isin(["true", "1"])
    return series.astype(bool)


def main():
    path = config.RESULTS_DIR / "raw_results.csv"
    if not path.exists():
        print("No results yet -- run `python run.py ...` first.")
        return

    df = pd.read_csv(path)
    df["truthful"] = _to_bool(df["truthful"])
    df["hallucinated"] = ~df["truthful"]

    pivot = (df.groupby(["category", "model"])["hallucinated"]
             .mean().unstack("model").sort_index())

    height = max(4, 0.45 * len(pivot))
    ax = pivot.plot(kind="barh", figsize=(9, height))
    ax.set_xlabel("Hallucination rate (lower is better)")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("Hallucination rate by topic category")
    ax.legend(title="model")
    plt.tight_layout()

    out = config.RESULTS_DIR / "hallucination_by_category.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()