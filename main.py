"""Top-level CLI for the platform.

Commands:
  evaluate        - run evaluations for models
  compare         - analyze existing results CSVs
  dashboard       - launch a simple visualization (placeholder)
  benchmark       - run the reproducible no-API local detector benchmark
  validate-judge  - check judge agreement with human labels (Cohen's kappa)
  validate-framework - check groundedness/risk engines vs human labels
  monitor         - stream one answer and track hallucination risk in real time
  list-datasets   - list benchmark datasets in the suite and their scoring
  bench           - run a benchmark (mmlu/gsm8k/bbq/halueval/toxicity) on a model

"""
import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
load_dotenv()

from src import benchmark_data as datasets, pipeline, analyze
import config
from src.visualization import plot_hallucination_by_category


def cmd_evaluate(args: argparse.Namespace):
    models = args.models or [config.MODEL_NAME]
    limit = args.limit if args.limit is not None else config.DEFAULT_LIMIT
    items = datasets.load_truthfulqa(limit=limit)
    print(f"Loaded {len(items)} questions; evaluating: {models}")
    all_results = []
    for m in models:
        all_results.extend(pipeline.evaluate_model(m, items))
    analyze.summarize(all_results)


def cmd_compare(args: argparse.Namespace):
    # Read saved CSVs if present and print summaries
    print("Comparing results from results/ directory...")
    csv = Path("results") / "raw_results.csv"
    if not csv.exists():
        print("No results/raw_results.csv found. Run 'evaluate' first.")
        return
    out = plot_hallucination_by_category(csv)
    print(f"Saved plot to {out}")


def cmd_dashboard(args: argparse.Namespace):
    import subprocess
    app = Path(__file__).parent / "app" / "app.py"
    print(f"Launching Streamlit dashboard ({app}). Press Ctrl+C to stop.")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)
    except FileNotFoundError:
        print("Streamlit not found. Install it with: pip install streamlit")


def cmd_benchmark(args: argparse.Namespace):
    """Run the reproducible, no-API local detector benchmark."""
    import subprocess
    script = Path(__file__).parent / "scripts" / "benchmark_local.py"
    subprocess.run([sys.executable, str(script)], check=False)


def cmd_monitor(args: argparse.Namespace):
    """Stream an answer and track hallucination risk in real time (Step 11)."""
    from src import risk

    last_level = {"v": None}

    def on_update(snap):
        # Print a line only when the risk band changes, to keep output readable.
        if snap["risk_level"] != last_level["v"]:
            last_level["v"] = snap["risk_level"]
            print(f"  [{snap['chars']:>4} chars] risk={snap['risk_score']:.2f} "
                  f"-> {snap['risk_level']}")

    print(f"Streaming {args.model} for: {args.question!r}\n")
    result = risk.stream_and_monitor(args.model, args.question, on_update=on_update)
    print("\n--- answer ---")
    print(result["final_text"])
    fr = result["final_risk"]
    print(f"\nfinal risk: {fr['risk_score']:.2f} ({fr['risk_level']})")
    actions = result["recommended"].get("recommended_interventions", [])
    print(f"recommended interventions: {actions or 'none'}")


def cmd_list_datasets(args: argparse.Namespace):
    print("Benchmark suite (name -> task / scoring):")
    for name in datasets.available():
        print(f"  {name:12} -> {datasets.TASKS.get(name, '?')}")
    print("\nTruthfulQA is the validated core. The others download from "
          "HuggingFace on first use; toxicity needs a real classifier backend "
          "for meaningful numbers (the built-in scorer is a lexical placeholder).")


def cmd_bench(args: argparse.Namespace):
    """Run a non-TruthfulQA benchmark end-to-end on a model."""
    from src import models
    def gen(prompt: str) -> str:
        return models.chat(args.model, prompt)

    tox_backend = None
    if args.name == "toxicity":
        from src.evaluators.toxicity_backends import get_backend
        # FAIL FAST: no silent fallback to the lexical scanner during a benchmark.
        try:
            tox_backend = get_backend(args.toxicity_backend)
        except ImportError as e:
            print("\n" + "!" * 70)
            print(str(e))
            print("!" * 70)
            sys.exit(1)
        if args.toxicity_backend == "lexical":
            print("  WARNING: lexical toxicity scorer is a weak placeholder — "
                  "these numbers are NOT validated.")
        print(f"  toxicity backend: {args.toxicity_backend}")

    print(f"Running '{args.name}' on {args.model} "
          f"(limit={args.limit}) ... downloads from HuggingFace on first use.")
    result = datasets.run_benchmark(args.name, gen, limit=args.limit,
                                    toxicity_backend=tox_backend)
    print(f"\n{args.name}: n={result['n']} task={result['task']}")
    if "accuracy" in result:
        print(f"accuracy: {result['accuracy']:.3f}  ({result['scored']} scored)")
    if "mean_toxicity" in result:
        print(f"mean toxicity: {result['mean_toxicity']:.3f}  "
              f"toxic rate: {result['toxic_rate']:.3f}  "
              f"(lexical placeholder unless a backend is wired)")


def cmd_validate_judge(args: argparse.Namespace):
    from src.research.judge_validation import validate_judge
    if not Path(args.labels).exists():
        print(f"Labels CSV not found: {args.labels}")
        print("Expected columns: question, answer, human_truthful "
              "[, correct_answers, incorrect_answers] (refs '||'-separated).")
        return
    validate_judge(args.labels, judge_model=args.judge_model)


def cmd_validate_framework(args: argparse.Namespace):
    from src.research.framework_validation import validate_framework
    if not Path(args.labels).exists():
        print(f"Labels CSV not found: {args.labels}")
        print("Expected columns: response, context, factual_groundedness, risk_level")
        return
    validate_framework(args.labels)


def main():
    ap = argparse.ArgumentParser(prog="main.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("evaluate")
    p.add_argument("--models", nargs="+", help="Model names to evaluate")
    p.add_argument("--limit", type=int, help="Limit number of questions")

    sub.add_parser("compare")
    sub.add_parser("dashboard")
    sub.add_parser("benchmark")

    vj = sub.add_parser("validate-judge",
                        help="Check judge agreement with human labels (Cohen's kappa)")
    vj.add_argument("--labels", required=True,
                    help="CSV of hand-labelled answers (question, answer, human_truthful)")
    vj.add_argument("--judge-model", default=None,
                    help="Override the judge model (defaults to config.JUDGE_MODEL)")

    vf = sub.add_parser("validate-framework",
                        help="Check groundedness/risk engines vs human labels "
                             "(confusion matrix + Pearson)")
    vf.add_argument("--labels", default="data/validation_human_labels.csv",
                    help="CSV: response, context, factual_groundedness, risk_level")

    mon = sub.add_parser("monitor",
                        help="Stream one answer and track hallucination risk live")
    mon.add_argument("--model", default=config.MODEL_NAME, help="Model to stream")
    mon.add_argument("--question", required=True, help="Question to ask")

    sub.add_parser("list-datasets", help="List benchmark datasets and their scoring")

    bench = sub.add_parser("bench", help="Run a benchmark (mmlu/gsm8k/bbq/halueval/toxicity)")
    bench.add_argument("name", help="Benchmark name (see list-datasets)")
    bench.add_argument("--model", default=config.MODEL_NAME, help="Model to evaluate")
    bench.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT,
                       help="Number of examples")
    bench.add_argument("--toxicity-backend", default="detoxify",
                       choices=["detoxify", "lexical"],
                       help="Toxicity scorer (detoxify = real classifier; falls "
                            "back to lexical if not installed)")

    args = ap.parse_args()
    if args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "dashboard":
        cmd_dashboard(args)
    elif args.cmd == "benchmark":
        cmd_benchmark(args)
    elif args.cmd == "validate-judge":
        cmd_validate_judge(args)
    elif args.cmd == "validate-framework":
        cmd_validate_framework(args)
    elif args.cmd == "monitor":
        cmd_monitor(args)
    elif args.cmd == "list-datasets":
        cmd_list_datasets(args)
    elif args.cmd == "bench":
        cmd_bench(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

