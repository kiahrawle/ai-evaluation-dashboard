from src.evaluators.groundedness import extract_claims, verify_citation_support


def test_extract_claims():
    answer = "The moon orbits Earth. It takes 27 days. The distance is 384,400 km. This is important."
    claims = extract_claims(answer)
    assert len(claims) > 0
    assert any("orbits" in c.lower() for c in claims)


def test_verify_citation_support():
    docs = [
        {"text": "The moon orbits the Earth at an average distance of 384,400 km."},
        {"text": "The moon takes approximately 27.3 days to orbit the Earth."}
    ]
    answer = "The moon orbits Earth and takes about 27 days."
    result = verify_citation_support(answer, docs)
    assert "citation_supported" in result
    assert "support_score" in result
    assert result["total_claims"] > 0
    assert result["supported_claims"] >= 0


def test_verify_citation_support_no_docs():
    result = verify_citation_support("Some answer", [])
    assert result["citation_supported"] is False
    assert result["support_score"] == 0.0
