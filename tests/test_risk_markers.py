from src.utils.risk_markers import (
    detect_uncertainty,
    detect_risky_phrases,
    detect_contradictions,
    detect_citation_gaps,
    scan_for_risk_markers,
)


def test_detect_uncertainty():
    text = "I think the answer might be around 42, but I'm not sure."
    result = detect_uncertainty(text)
    assert result["uncertainty_score"] > 0.3
    assert len(result["uncertainty_phrases"]) > 0


def test_detect_risky_phrases():
    text = "As far as I know, I can't verify this, and I'm making an educated guess."
    result = detect_risky_phrases(text)
    assert result["risky_score"] > 0.2
    assert len(result["risky_phrases"]) > 0


def test_detect_contradictions():
    text = "The earth is flat. Actually, wait, I meant the earth is round."
    result = detect_contradictions(text)
    assert result["contradiction_score"] > 0.0
    assert len(result["contradiction_markers"]) > 0


def test_detect_citation_gaps():
    text = "According to studies, the sky is purple."
    result = detect_citation_gaps(text)
    assert result["citation_gap_score"] > 0.0  # has claim but no URL/bracket citations
    assert result["has_claims_needing_citations"] is True


def test_scan_for_risk_markers():
    text = "I think that according to some research, the answer is probably 42."
    result = scan_for_risk_markers(text)
    assert "uncertainty" in result
    assert "risky_phrases" in result
    assert "contradictions" in result
    assert "citation_gaps" in result
