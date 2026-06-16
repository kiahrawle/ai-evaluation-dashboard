"""Reproducible, no-API benchmark of the deterministic local detectors.

This does NOT measure any model's hallucination rate (that needs API generation
+ a judge). It measures the platform's own rule-based components against a small
hand-labelled fixture, so the numbers are exact and reproducible by anyone:

    python scripts/benchmark_local.py

Outputs a table to stdout, writes docs/benchmark_local.json, and renders
docs/benchmark_local.png.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.evaluators.refusal_quality import detect_is_refusal
from src.evaluators.groundedness import detect_numeric_contradictions


def _metrics(preds, labels):
    n = len(labels)
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    tn = sum(1 for p, l in zip(preds, labels) if not p and not l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"n": n, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def run() -> dict:
    cases = json.loads((ROOT / "data" / "benchmark_cases.json").read_text(encoding="utf-8"))

    refusal = cases["refusal_detection"]
    r_preds = [detect_is_refusal(c["text"]) for c in refusal]
    r_labels = [c["label"] for c in refusal]

    contra = cases["numeric_contradiction"]
    c_preds = [detect_numeric_contradictions(c["answer"], [{"text": c["doc"]}])["contradiction_detected"]
               for c in contra]
    c_labels = [c["label"] for c in contra]

    return {
        "refusal_detection": _metrics(r_preds, r_labels),
        "numeric_contradiction": _metrics(c_preds, c_labels),
    }


def _save_chart(results: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping chart)")
        return
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    import numpy as np
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, met in enumerate(metrics):
        ax.bar(x + i * width, [results[n][met] for n in names], width, label=met)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Local detector benchmark (no-API, reproducible)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart -> {out_path}")


def main():
    results = run()
    print("\n=== Local detector benchmark (no API) ===")
    header = f"{'component':24}{'n':>4}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:24}{m['n']:>4}{m['accuracy']:>8.2f}{m['precision']:>8.2f}"
              f"{m['recall']:>8.2f}{m['f1']:>8.2f}")

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "benchmark_local.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics -> {docs / 'benchmark_local.json'}")
    _save_chart(results, docs / "benchmark_local.png")


if __name__ == "__main__":
    main()
