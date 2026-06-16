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
| Numeric contradiction | 8 | 1.00 | 1.00 | 1.00 | 1.00 |

![Local detector benchmark](docs/benchmark_local.png)

**How numeric contradiction works (and why recall went 0.75 → 1.00):** the first
version gated on shared *context words* before comparing numbers, so it missed
*"90% of patients"* vs *"45% of patients"* — same quantity, different wording
("symptoms" vs "improvement"). The detector now embeds the number-bearing
sentences with `all-MiniLM-L6-v2` and compares numbers only when the sentences
are *semantically* similar (cosine ≥ 0.45, tuned for this model — short factual
sentences score ~0.5-0.75 for the same fact vs ~0.2-0.35 across topics, so a
naive 0.75 cutoff would actually miss most real contradictions). Precision stays
1.00 on the fixture: topically-unrelated numbers (revenue vs population, price vs
temperature) fall below the threshold and aren't compared.

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

Run one with `python main.py bench mmlu --model <m> --limit 100`. Loaders are
**local-first**: they read `data/cached_benchmarks/` before touching HuggingFace.
Cache the datasets once for offline / CI use:

```bash
python scripts/download_benchmarks.py            # caches mmlu, gsm8k, bbq
python scripts/download_benchmarks.py --datasets mmlu gsm8k bbq halueval toxicity --limit 1000
```

A missing cache file triggers a one-off download with a warning. The cache dir
is gitignored, so restore it from a CI cache step or run the script once. The scoring logic (choice parsing, numeric
extraction, detection, toxicity) is unit-tested offline, but end-to-end numbers
are not committed (they depend on the model).

Toxicity uses a **real classifier** (`detoxify`, the default backend, installed
by default): it downloads model weights once, then scores continuations locally
on CPU. If `detoxify` is missing the toxicity benchmark **fails fast** with a
critical error instead of silently scoring with a weak lexical scanner — so a
formal run can never report misleadingly "clean" numbers by accident. The
lexical scorer is available only via an explicit `--toxicity-backend lexical`,
which prints an "unvalidated" warning.

## 4. Framework validation (groundedness + risk vs. human labels)

The groundedness and risk engines are roadmap layers, so they get the same
treatment as the judge: checked against human labels. `data/validation_human_labels.csv`
hand-labels 40 (response, context) pairs with `factual_groundedness` (0/1) and
`risk_level` (LOW/MEDIUM/HIGH); the validator runs the real engines and reports
confusion matrices + Pearson correlation.

```bash
python main.py validate-framework        # uses data/validation_human_labels.csv
```

Latest run on the 40-row set (deterministic — embeddings, no API):

| Engine | Result |
|---|---|
| Groundedness (binary) | accuracy 0.60, precision 0.62, recall 0.83, F1 0.71 (TP 20 / TN 4 / FP 12 / FN 4) |
| Risk band (3-class) | accuracy 0.55 |
| Risk score vs human ordinal | Pearson **0.322** (weak) |

**What this tells us — and why it's worth having:** the groundedness engine has
decent recall but poor precision (12 false positives) — embedding similarity to
the context stays high even when the response *contradicts* it, so it over-marks
answers as "supported." The risk engine is weakly correlated (0.32) and, on this
set, predicted **no** HIGH-risk rows: confident fabrications carry no uncertainty
markers, so the heuristic risk score misses them. These are real limitations, not
tuning noise — concrete evidence for why those layers are labelled *roadmap*, and
a baseline to improve against (e.g. NLI-based contradiction for groundedness,
reference-aware signals for risk). Expand the CSV with your own labels to harden
the numbers.

## Reproducing the figures

| Artifact | Command |
|---|---|
| Local detector table + chart | `python scripts/benchmark_local.py` |
| Per-category hallucination chart | `python plot.py` (after an `evaluate` run) |
| Live dashboard | `streamlit run app/app.py` |
