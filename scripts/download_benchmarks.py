"""Download benchmark datasets once and cache them locally as JSON records.

Relying on HuggingFace at every run is slow and breaks offline / CI use. Run
this once to populate data/cached_benchmarks/; the loaders then read locally.

    python scripts/download_benchmarks.py                 # mmlu, gsm8k, bbq
    python scripts/download_benchmarks.py --datasets mmlu gsm8k bbq halueval toxicity
    python scripts/download_benchmarks.py --limit 1000     # more rows per dataset

Each dataset is serialized as normalized records (identical to what the loader
returns) at data/cached_benchmarks/<name>__<split>.json.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.benchmark_data import mmlu, gsm8k, bbq, halueval, toxicity, cache

MODULES = {
    "mmlu": mmlu,
    "gsm8k": gsm8k,
    "bbq": bbq,
    "halueval": halueval,
    "toxicity": toxicity,
}
DEFAULT = ["mmlu", "gsm8k", "bbq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT, choices=list(MODULES),
                    help="Which benchmarks to cache (default: mmlu gsm8k bbq)")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max records to cache per dataset (default: 500)")
    args = ap.parse_args()

    failures = []
    for name in args.datasets:
        mod = MODULES[name]
        split = getattr(mod, "SPLIT", "test")
        print(f"Downloading '{name}' (split={split}, up to {args.limit} records) ...")
        try:
            records = mod._download(args.limit, split)
        except Exception as e:  # network / missing config / mirror path
            print(f"  FAILED: {e}")
            failures.append(name)
            continue
        path = cache.save_cached(name, split, records)
        print(f"  cached {len(records)} records -> {path.relative_to(ROOT)}")

    if failures:
        print(f"\nDone with errors. Failed: {failures}. "
              f"(BBQ's HF path varies by mirror — adjust HF_PATH in "
              f"src/benchmark_data/bbq.py if it failed.)")
        sys.exit(1)
    print("\nDone. Loaders will now read these from data/cached_benchmarks/ first.")


if __name__ == "__main__":
    main()
