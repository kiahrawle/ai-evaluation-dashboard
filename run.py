"""Smoke-test runner: a thin wrapper around `python main.py evaluate`.

Kept only for the short command form. The actual evaluate/score/summarize logic
lives in main.cmd_evaluate so there is one implementation, not two.

    python run.py --models claude-haiku-4-5-20251001 --limit 20
"""
import argparse

from dotenv import load_dotenv
load_dotenv()  # must run before importing code that creates the API client

import main


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model names to evaluate, e.g. claude-haiku-4-5-20251001")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N questions (for smoke tests)")
    args = ap.parse_args()
    main.cmd_evaluate(args)


if __name__ == "__main__":
    _main()
