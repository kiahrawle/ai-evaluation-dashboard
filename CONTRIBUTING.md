# Contributing

Thanks for your interest. This project has a deliberate shape (see "Design
philosophy" in the README): a **small validated core** plus a clearly-labelled
**roadmap** of platform layers. Contributions are most welcome when they either
harden the core or move a roadmap layer toward being validated.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY (OPENAI_API_KEY optional)
```

## Tests

Unit tests mock external calls, so the suite runs offline:

```bash
python -m pytest -q
```

Every PR should keep the suite green and add tests for new logic. Pure logic
(scorers, normalizers, metrics) must be unit-tested without network or API
calls — follow the existing pattern of separating a pure function (e.g. a
dataset `_normalize`) from the I/O around it (the HF fetch).

## Conventions

- Match the surrounding code's style, naming, and comment density.
- Keep provider-specific code isolated (generation in `src/models/`, judge
  routing in `src/scoring.py`).
- New benchmarks: add a loader that normalizes to the registry schema, register
  it in `src/datasets/registry.py`, and add a scorer in
  `src/evaluators/tasks.py` if the task type is new.

## Honesty rule

Don't describe a feature as validated unless there's evidence for it. If a
metric isn't checked against human labels or a reference implementation, label
it as a heuristic/roadmap item in the docs. Scoped claims are the point.

## Pull requests

- Branch off `main`, keep PRs focused, and describe what's validated vs. not.
- Run `python -m pytest -q` and `python scripts/benchmark_local.py` before
  opening the PR.
