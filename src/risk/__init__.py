from typing import Dict, Any


def score_risk(
    hallucination_score: float,
    grounded_score: float,
    confidence_score: float | None = None,
    unsupported_claims_ratio: float | None = None,
    contradiction_score: float | None = None,
) -> float:
    """Weighted risk scoring formula.
    
    risk_score =
        hallucination_probability * 0.4 +
        unsupported_claims * 0.3 +
        (1 - confidence_level) * 0.2 +
        contradiction_score * 0.1
    """
    # Base: hallucination probability (40% weight)
    halluc_component = (hallucination_score or 0.0) * 0.4
    
    # Unsupported claims (30% weight)
    unsupported = (unsupported_claims_ratio or (1.0 - (grounded_score or 0.0))) * 0.3
    
    # Confidence (20% weight) - inverted (low confidence = high risk)
    conf = confidence_score or 0.5
    confidence_component = (1.0 - conf) * 0.2
    
    # Contradiction markers (10% weight)
    contradiction_component = (contradiction_score or 0.0) * 0.1
    
    risk = halluc_component + unsupported + confidence_component + contradiction_component
    return float(max(0.0, min(1.0, risk)))


def score_risk_with_details(
    hallucination_score: float,
    grounded_score: float,
    confidence_score: float | None = None,
    unsupported_claims_ratio: float | None = None,
    contradiction_score: float | None = None,
) -> Dict[str, Any]:
    """Return detailed risk breakdown with risk level classification."""
    risk = score_risk(
        hallucination_score,
        grounded_score,
        confidence_score,
        unsupported_claims_ratio,
        contradiction_score,
    )
    
    if risk < 0.33:
        level = "LOW"
    elif risk < 0.67:
        level = "MEDIUM"
    else:
        level = "HIGH"
    
    return {
        "risk_score": risk,
        "risk_level": level,
        "hallucination_score": hallucination_score,
        "grounded_score": grounded_score,
        "confidence_score": confidence_score or 0.5,
        "unsupported_claims_ratio": unsupported_claims_ratio,
        "contradiction_score": contradiction_score or 0.0,
    }


from .realtime import RealTimeRiskMonitor, stream_and_monitor, stream_risk_from_markers

__all__ = [
    "score_risk", "score_risk_with_details",
    "RealTimeRiskMonitor", "stream_and_monitor", "stream_risk_from_markers",
]
