from src import scoring


def test_evaluate_judges(monkeypatch):
    item = {
        "question": "Q?",
        "correct_answers": ["A"],
        "incorrect_answers": ["B"],
    }
    ans = "some answer"

    # avoid filesystem cache reads/writes
    monkeypatch.setattr(scoring, "_load_judge_cache", lambda model=None: {})
    monkeypatch.setattr(scoring, "_save_judge_cache", lambda cache, model=None: None)

    def fake_ask(prompt, model=None):
        if model and "a" in model:
            return {"truthful": True, "informative": True}
        else:
            return {"truthful": False, "informative": True}

    monkeypatch.setattr(scoring, "_ask_judge", fake_ask)

    res = scoring.evaluate_judges(item, ans, judges=["judge-a", "judge-b"]) 
    assert "verdicts" in res
    assert res["agreement_rate"] == 0.5
    assert res["disagreement_score"] == 0.5
