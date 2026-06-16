from src.evaluators.claims import decompose_claims, score_claims


def test_decompose_claims_splits_sentences():
    answer = "The moon orbits Earth. It takes about 27 days. Short."
    claims = decompose_claims(answer)
    # "Short." is < 10 chars after stripping punctuation and should be dropped.
    assert "The moon orbits Earth" in claims
    assert any("27 days" in c for c in claims)
    assert all(len(c) > 10 for c in claims)


def test_decompose_empty():
    assert decompose_claims("") == []
    assert decompose_claims("   ") == []


def test_score_claims_flags_contradiction():
    item = {
        "correct_answers": ["Chocolate can be toxic to dogs."],
        "incorrect_answers": ["Chocolate is completely safe for dogs to eat."],
    }
    answer = "Chocolate is completely safe for dogs to eat."
    out = score_claims(item, answer)
    assert out["total_claims"] >= 1
    assert out["contradicted"] >= 1
    assert out["claim_hallucination_rate"] > 0.0


def test_score_claims_supported():
    item = {
        "correct_answers": ["The Earth orbits the Sun."],
        "incorrect_answers": ["The Sun orbits the Earth."],
    }
    answer = "The Earth orbits the Sun."
    out = score_claims(item, answer)
    assert out["supported"] >= 1
    assert out["contradicted"] == 0


def test_score_claims_no_references():
    item = {"correct_answers": [], "incorrect_answers": []}
    out = score_claims(item, "Some factual sounding statement here.")
    assert out["total_claims"] >= 1
    assert out["unsupported"] == out["total_claims"]
