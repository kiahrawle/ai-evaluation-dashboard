from src.evaluators.self_consistency import measure_consistency, self_consistency_score


def test_identical_responses_are_consistent():
    out = measure_consistency(["The sky is blue."] * 4)
    assert out["n"] == 4
    assert out["consistency"] > 0.95
    assert out["hallucination_signal"] < 0.05


def test_divergent_responses_lower_consistency():
    same = measure_consistency([
        "Paris is the capital of France.",
        "The capital of France is Paris.",
    ])
    diverse = measure_consistency([
        "Paris is the capital of France.",
        "Bananas are a good source of potassium for athletes.",
    ])
    assert diverse["consistency"] < same["consistency"]
    assert diverse["hallucination_signal"] > same["hallucination_signal"]


def test_single_response_is_trivially_consistent():
    out = measure_consistency(["only one"])
    assert out["consistency"] == 1.0
    assert out["hallucination_signal"] == 0.0


def test_self_consistency_score_uses_injected_sampler():
    calls = {}

    def fake_sampler(model, question, n, temperature):
        calls["args"] = (model, question, n, temperature)
        return ["A is true."] * n

    out = self_consistency_score("m", "Is A true?", n=3, sampler=fake_sampler)
    assert calls["args"] == ("m", "Is A true?", 3, 0.7)
    assert out["n"] == 3
    assert "samples" in out and len(out["samples"]) == 3
