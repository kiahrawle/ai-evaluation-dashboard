import pytest


def test_pipeline_smoke(monkeypatch):
    from src import pipeline

    items = [
        {
            "question": "Is the sky blue?",
            "category": "Nature",
            "best_answer": "",
            "correct_answers": ["Yes, due to Rayleigh scattering."],
            "incorrect_answers": ["No, it's green."],
        }
    ]

    # Patch generate_all to avoid external API calls
    monkeypatch.setattr("src.models.generate_all", lambda model, items: {items[0]["question"]: "Yes, due to Rayleigh scattering."})

    # Patch scoring internals to simulate multiple judges
    monkeypatch.setattr("src.scoring.evaluate_judges", lambda item, ans, judges=None: {
        "verdicts": {"judge-a": {"truthful": True, "informative": True}, "judge-b": {"truthful": True, "informative": True}},
        "agreement_rate": 1.0,
        "disagreement_score": 0.0,
    })
    monkeypatch.setattr("src.scoring.semantic_truthful", lambda item, ans: True)

    # Patch rag.retrieve to return a supporting doc
    monkeypatch.setattr("src.rag.retriever.retrieve", lambda q, top_k=5: [{"text": "Rayleigh scattering causes the sky to appear blue."}])

    results = pipeline.evaluate_model("test-model", items)
    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    # Check expected keys from EvaluationResult dict
    assert r["prompt"] == items[0]["question"]
    assert r["response"].startswith("Yes")
    assert "hallucination_score" in r
