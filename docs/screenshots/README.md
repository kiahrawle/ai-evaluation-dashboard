# Screenshots & gallery

Two kinds of visuals live here:

1. **Generated chart artifacts** (committed): rendered by
   `python scripts/render_dashboard_charts.py` using the same `src/reporting.py`
   helpers the dashboard uses. By default they come from the illustrative
   `docs/sample_results.json` (titled "sample data"); after you run a real
   evaluation they regenerate from `results/results.json`:

   ```bash
   python main.py evaluate --models claude-haiku-4-5-20251001 --limit 50
   python scripts/render_dashboard_charts.py        # now uses real results
   ```

   - `risk_distribution.png`
   - `hallucination_by_category.png`
   - `../benchmark_local.png` — the reproducible no-API detector benchmark

2. **Live dashboard screenshots** (capture yourself): the Streamlit UI is
   interactive, so capture it with your OS screenshot tool:

   ```bash
   streamlit run app/app.py     # opens http://localhost:8501
   ```

   Suggested shots: Dashboard (metrics + charts), Live Evaluation (risk markers),
   Upload & Analyze (grounding result), Leaderboard. Save them here as
   `dashboard.png`, `live_eval.png`, etc., and reference them from the README.
