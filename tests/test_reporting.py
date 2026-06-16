import json

from src import reporting


def _result(model, truthful, risk, level, category, cite, dangerous=None, contra=False):
    return {
        "prompt": "q", "response": "a", "model": model,
        "truthful": truthful, "grounded": truthful, "risk_score": risk,
        "metadata": {
            "risk_level": level, "category": category,
            "citation_supported": cite, "dangerous_level": dangerous,
            "numeric_contradiction": contra,
        },
    }


SAMPLE = [
    _result("m1", True, 0.2, "LOW", "Health", True),
    _result("m1", False, 0.8, "HIGH", "Health", False, dangerous="HIGH", contra=True),
    _result("m2", False, 0.5, "MEDIUM", "Law", True),
]


def test_overview_metrics():
    m = reporting.overview_metrics(SAMPLE)
    assert m["total"] == 3
    assert abs(m["hallucination_rate"] - 2 / 3) < 1e-9
    assert abs(m["avg_risk_score"] - 0.5) < 1e-9
    # 2 of 3 have citation_supported True
    assert abs(m["citation_coverage"] - 2 / 3) < 1e-9
    assert m["dangerous_high"] == 1
    assert m["contradictions"] == 1


def test_overview_metrics_empty():
    m = reporting.overview_metrics([])
    assert m["total"] == 0 and m["hallucination_rate"] == 0.0


def test_risk_distribution():
    assert reporting.risk_distribution(SAMPLE) == {"LOW": 1, "MEDIUM": 1, "HIGH": 1}


def test_hallucination_by_category():
    by_cat = reporting.hallucination_by_category(SAMPLE)
    assert by_cat["Health"] == 0.5   # one true, one false
    assert by_cat["Law"] == 1.0


def test_leaderboard_from_results_ranks_by_hallucination():
    rows = reporting.leaderboard_from_results(SAMPLE)
    # m1: 1/2 = 0.5 hallucination, m2: 1/1 = 1.0 -> m1 ranks first (lower is better)
    assert [r["model"] for r in rows] == ["m1", "m2"]
    assert rows[0]["model"] == "m1"
    assert rows[0]["rank"] == 1


def test_load_results_roundtrip(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    loaded = reporting.load_results(p)
    assert len(loaded) == 3
    assert reporting.load_results(tmp_path / "missing.json") == []


def test_load_results_csv_fallback(tmp_path):
    import pandas as pd
    # No results.json, but a raw_results.csv with a stringified metadata dict.
    rows = [{
        "truthful": True, "grounded": True, "risk_score": 0.2, "model": "m1",
        "category": "Health",
        "metadata": str({"risk_level": "LOW", "category": "Health",
                         "citation_supported": True}),
    }]
    pd.DataFrame(rows).to_csv(tmp_path / "raw_results.csv", index=False)
    loaded = reporting.load_results(tmp_path / "results.json")  # missing -> CSV fallback
    assert len(loaded) == 1
    assert loaded[0]["truthful"] is True
    assert loaded[0]["metadata"]["risk_level"] == "LOW"
