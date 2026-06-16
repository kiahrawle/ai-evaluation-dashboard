from src.risk import score_risk, score_risk_with_details
from src.guardrails.interventions import (
    classify_risk_level,
    recommend_interventions,
    apply_interventions,
    RiskLevel,
)


def test_score_risk_weighted():
    # High hallucination
    risk = score_risk(
        hallucination_score=0.8,
        grounded_score=0.2,
        confidence_score=0.3,
        contradiction_score=0.4,
    )
    assert risk > 0.5  # Should be HIGH risk


def test_score_risk_low():
    # Low risk all around
    risk = score_risk(
        hallucination_score=0.1,
        grounded_score=0.9,
        confidence_score=0.9,
        contradiction_score=0.0,
    )
    assert risk < 0.2  # Should be LOW risk


def test_score_risk_with_details():
    details = score_risk_with_details(
        hallucination_score=0.7,
        grounded_score=0.3,
        confidence_score=0.4,
        contradiction_score=0.3,
    )
    assert "risk_score" in details
    assert "risk_level" in details
    assert details["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_classify_risk_level():
    assert classify_risk_level(0.2) == RiskLevel.LOW
    assert classify_risk_level(0.5) == RiskLevel.MEDIUM
    assert classify_risk_level(0.8) == RiskLevel.HIGH


def test_recommend_interventions_high_risk():
    risk_info = {
        "risk_level": "HIGH",
        "hallucination_score": 0.8,
        "citation_supported": False,
        "contradiction_score": 0.5,
        "uncertainty_score": 0.3,
        "citation_gap_score": 0.7,
    }
    result = recommend_interventions(risk_info)
    assert len(result["recommended_interventions"]) > 0
    assert "warn_user" in result["recommended_interventions"]


def test_recommend_interventions_low_risk():
    risk_info = {
        "risk_level": "LOW",
        "hallucination_score": 0.1,
        "citation_supported": True,
        "contradiction_score": 0.0,
        "uncertainty_score": 0.1,
        "citation_gap_score": 0.0,
    }
    result = recommend_interventions(risk_info)
    # Low risk should have minimal or no interventions
    assert len(result["recommended_interventions"]) == 0


def test_apply_interventions():
    response = "The answer is 42."
    interventions = ["warn_user", "force_citations"]
    result = apply_interventions(interventions, response)
    assert result["original_response"] == response
    assert len(result["warnings"]) > 0
    assert len(result["instructions"]) > 0
