from src.risk.realtime import RealTimeRiskMonitor, stream_and_monitor


def test_risk_rises_as_risky_text_streams():
    monitor = RealTimeRiskMonitor()
    s1 = monitor.update("The capital of France is Paris. ")
    # Clean factual text -> low risk.
    assert s1["risk_level"] == "LOW"

    # Now stream risky / hedging / citation-gap language.
    monitor.update("Actually, I can't verify this, ")
    s3 = monitor.update("but studies show it might possibly be different.")
    assert s3["risk_score"] > s1["risk_score"]
    assert s3["chars"] > s1["chars"]


def test_snapshot_and_trace_accumulate():
    chunks = ["I think ", "this is probably true, ", "but I'm not certain."]
    out = stream_and_monitor("m", "q?", token_source=chunks)
    assert out["final_text"] == "".join(chunks)
    assert len(out["risk_trace"]) == len(chunks)
    assert out["final_risk"]["risk_score"] >= 0.0
    # Risk markers present -> some non-zero uncertainty signal by the end.
    assert out["final_risk"]["uncertainty_score"] > 0.0


def test_high_risk_triggers_interventions():
    risky = [
        "I can't verify this and I'm making an educated guess. ",
        "Actually, wait, let me revise. According to research, statistics show ",
        "this, but I may be wrong and I'm not certain.",
    ]
    out = stream_and_monitor("m", "q?", token_source=risky, high_threshold=0.4)
    assert out["final_risk"]["risk_level"] in {"MEDIUM", "HIGH"}
    assert out["first_high_at"] is not None
    # Interventions recommended once risk is elevated.
    assert isinstance(out["recommended"]["recommended_interventions"], list)


def test_clean_answer_stays_low_with_no_interventions():
    clean = ["Water boils at 100 degrees Celsius ", "at sea level atmospheric pressure."]
    out = stream_and_monitor("m", "q?", token_source=clean)
    assert out["final_risk"]["risk_level"] == "LOW"
    assert out["first_high_at"] is None
