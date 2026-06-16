from src.evaluators.groundedness import (
    detect_numeric_contradictions,
    verify_citation_support,
)


def test_revenue_contradiction_detected():
    docs = [{"text": "In fiscal year 2023 our annual revenue was 12 million dollars."}]
    answer = "The company's revenue was 50 million dollars."
    out = detect_numeric_contradictions(answer, docs)
    assert out["contradiction_detected"] is True
    assert out["num_contradictions"] >= 1
    assert "revenue" in out["contradictions"][0]["context"]


def test_close_numbers_not_contradiction():
    docs = [{"text": "The moon takes approximately 27.3 days to orbit the Earth."}]
    answer = "The moon takes about 27 days to orbit the Earth."
    out = detect_numeric_contradictions(answer, docs)
    assert out["contradiction_detected"] is False


def test_no_shared_context_no_contradiction():
    docs = [{"text": "The population of the city is 8 million."}]
    answer = "The revenue was 50 million dollars."
    out = detect_numeric_contradictions(answer, docs)
    # Different subjects (population vs revenue) -> not a contradiction.
    assert out["contradiction_detected"] is False


def test_paraphrased_subject_still_caught():
    # Same quantity, different surrounding words. The old word-overlap detector
    # missed this; the embedding-gated one catches it.
    docs = [{"text": "The trial reported improvement in 45 percent of patients."}]
    answer = "The drug reduced symptoms in 90 percent of patients."
    out = detect_numeric_contradictions(answer, docs)
    assert out["contradiction_detected"] is True
    assert out["contradictions"][0]["similarity"] >= 0.45


def test_citation_support_links_evidence():
    docs = [
        {"text": "The moon orbits the Earth at an average distance of 384,400 km."},
        {"text": "The moon takes approximately 27.3 days to orbit the Earth."},
    ]
    answer = "The moon orbits Earth and takes about 27 days."
    result = verify_citation_support(answer, docs)
    assert "evidence" in result
    assert len(result["evidence"]) == result["total_claims"]
    for ev in result["evidence"]:
        assert "claim" in ev and "supported" in ev and "similarity" in ev
        if ev["supported"]:
            assert ev["doc_index"] in (0, 1)
            assert ev["evidence_text"] is not None
