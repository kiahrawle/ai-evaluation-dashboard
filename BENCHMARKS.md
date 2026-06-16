# Benchmarks

Two very different things get called "benchmarks" in an eval project. We keep
them separate and honest.

## 1. Local detector benchmark (reproducible, no API)

These measure the platform's own **rule-based detectors** against a small
hand-labelled fixture ([data/benchmark_cases.json](data/benchmark_cases.json)).
They use no model generation and no judge, so anyone gets identical numbers:

```bash
python scripts/benchmark_local.py
```

Latest run:

| Component | n | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Refusal detection | 8 | 1.00 | 1.00 | 1.00 | 1.00 |
| Numeric contradiction | 8 | 0.88 | 1.00 | 0.75 | 0.86 |

![Local detector benchmark](docs/benchmark_local.png)

**Reading this honestly:** numeric-contradiction detection has perfect
precision on the fixture (no false alarms) but recall 0.75 — it misses one true
contradiction (*"90% of patients"* vs *"45% of patients"*) because the answer
and document describe the same quantity with different surrounding words
("symptoms" vs "improvement") and the detector requires a shared context word to
compare two numbers. That is a known limitation of the lexical-context approach,
not a tuning artifact. Embedding-aligned context matching would raise recall.

## 2. Model hallucination benchmark (requires API keys)

This is the real TruthfulQA evaluation: generate answers with a model, judge
them, and report the hallucination rate with confidence intervals. It needs
`ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` for GPT judges/models), so the numbers
depend on your account and are not committed here.

```bash
# Evaluate one or more models over the first N TruthfulQA questions
python main.py evaluate --models claude-haiku-4-5-20251001 --limit 100

# Results land in results/ (overall.csv, comparison.csv, results.json) and the
# Streamlit dashboard + leaderboard read results/results.json automatically.
streamlit run app/app.py
```

Reported metrics per model:

- **Hallucination rate** — share of answers the judge marks *not truthful*
  (Wilson 95% CI in `overall.csv`).
- **Truthful-and-informative rate** — guards against a model gaming truthfulness
  by refusing everything.
- **Groundedness / citation coverage** — when a RAG corpus is loaded.
- **Avg risk score** — weighted risk-engine output.

### Validate the judge first

Before trusting any model numbers, check the judge against your own labels:

```bash
python main.py validate-judge --labels your_labels.csv
```

It reports raw agreement and Cohen's kappa. If kappa is weak (< 0.4), fix the
judge prompt before reporting anything.

## 3. Broader benchmark suite

Beyond TruthfulQA, the suite wires in additional benchmarks as normalized
loaders + task-appropriate scorers (`python main.py list-datasets`):

| Dataset | Task | Metric |
|---|---|---|
| HaluEval | hallucination detection | detection accuracy |
| MMLU | 4-way multiple choice | accuracy |
| GSM8K | numeric math | exact-match accuracy |
| BBQ | multiple choice (social bias) | accuracy (bias score = roadmap) |
| RealToxicityPrompts | prompt continuation | mean toxicity / toxic rate |

Run one with `python main.py bench mmlu --model <m> --limit 100`. These download
from HuggingFace on first use. The scoring logic (choice parsing, numeric
extraction, detection, toxicity) is unit-tested offline, but end-to-end numbers
are not committed (they depend on the model).

Toxicity uses a **real classifier** when `detoxify` is installed (the default
`--toxicity-backend detoxify`): it downloads model weights once, then scores
continuations locally on CPU. Without it, scoring falls back to a lexical
placeholder (flagged in the output) — so trust toxicity numbers only with the
detoxify backend active.

## Reproducing the figures

| Artifact | Command |
|---|---|
| Local detector table + chart | `python scripts/benchmark_local.py` |
| Per-category hallucination chart | `python plot.py` (after an `evaluate` run) |
| Live dashboard | `streamlit run app/app.py` |
