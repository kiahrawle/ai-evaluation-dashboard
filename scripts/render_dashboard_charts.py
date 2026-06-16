"""Render the dashboard's data panels to PNGs for the README gallery.

Uses the same reporting helpers the live Streamlit app uses, so the charts
reflect real computation. Defaults to results/results.json; falls back to the
committed illustrative sample (docs/sample_results.json) so the gallery has
images before you run an evaluation.

    python scripts/render_dashboard_charts.py [path/to/results.json]

Writes docs/screenshots/risk_distribution.png and hallucination_by_category.png.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import reporting

RISK_COLORS = {"LOW": "#00CC96", "MEDIUM": "#FFA15A", "HIGH": "#FF6B6B"}
OUT = ROOT / "docs" / "screenshots"


def _pick_source(argv) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    live = ROOT / "results" / "results.json"
    return live if live.exists() else (ROOT / "docs" / "sample_results.json")


def main():
    src = _pick_source(sys.argv)
    results = reporting.load_results(src)
    if not results:
        print(f"No results found at {src}.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    is_sample = src.name == "sample_results.json"
    tag = " (sample data)" if is_sample else ""

    # Risk-level distribution (pie).
    dist = reporting.risk_distribution(results)
    labels = [k for k, v in dist.items() if v]
    sizes = [dist[k] for k in labels]
    if sizes:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=labels, autopct="%1.0f%%",
               colors=[RISK_COLORS[k] for k in labels], startangle=90)
        ax.set_title(f"Risk level distribution{tag}")
        fig.tight_layout()
        fig.savefig(OUT / "risk_distribution.png", dpi=150)
        print(f"wrote {OUT / 'risk_distribution.png'}")

    # Hallucination by category (bar).
    by_cat = reporting.hallucination_by_category(results)
    if by_cat:
        items = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
        cats = [c for c, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(cats, vals, color="#636EFA")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Hallucination rate")
        ax.set_title(f"Hallucination by category{tag}")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(OUT / "hallucination_by_category.png", dpi=150)
        print(f"wrote {OUT / 'hallucination_by_category.png'}")


if __name__ == "__main__":
    main()
