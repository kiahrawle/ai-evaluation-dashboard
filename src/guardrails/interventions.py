"""Risk interventions: actions taken when risk is detected."""
from typing import Dict, Any, List
from enum import Enum


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class InterventionAction(str, Enum):
    """Available intervention actions."""
    WARN_USER = "warn_user"
    FORCE_CITATIONS = "force_citations"
    ACTIVATE_STRICT_GUARDRAIL = "activate_strict_guardrail"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    SWITCH_MODEL = "switch_model"


def classify_risk_level(risk_score: float) -> RiskLevel:
    """Classify risk score into LOW/MEDIUM/HIGH."""
    if risk_score < 0.33:
        return RiskLevel.LOW
    elif risk_score < 0.67:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH


def recommend_interventions(risk_info: Dict[str, Any]) -> Dict[str, Any]:
    """Recommend interventions based on risk analysis.
    
    Input: {
        "risk_score": float,
        "risk_level": str,
        "hallucination_score": float,
        "citation_supported": bool,
        "uncertainty_score": float,
        "contradiction_score": float,
        "citation_gap_score": float
    }
    
    Returns: {
        "recommended_interventions": List[InterventionAction],
        "rationale": Dict[str, str],
        "severity": str
    }
    """
    interventions = []
    rationale = {}
    
    risk_level = risk_info.get("risk_level", RiskLevel.LOW.value)
    halluc = risk_info.get("hallucination_score", 0.0) or 0.0
    citation_supported = risk_info.get("citation_supported", True)
    uncertainty = risk_info.get("uncertainty_score", 0.0) or 0.0
    contradiction = risk_info.get("contradiction_score", 0.0) or 0.0
    citation_gap = risk_info.get("citation_gap_score", 0.0) or 0.0
    
    # HIGH risk interventions
    if risk_level == RiskLevel.HIGH.value:
        if halluc > 0.7:
            interventions.append(InterventionAction.ACTIVATE_STRICT_GUARDRAIL)
            rationale["strict_guardrail"] = f"High hallucination probability ({halluc:.2%})"
        if citation_gap > 0.5 or not citation_supported:
            interventions.append(InterventionAction.FORCE_CITATIONS)
            rationale["citations"] = "Answer contains claims lacking citation support"
        if contradiction > 0.4:
            interventions.append(InterventionAction.ASK_CLARIFYING_QUESTION)
            rationale["clarification"] = "Multiple contradictions detected; user clarification needed"
        interventions.append(InterventionAction.WARN_USER)
        rationale["warning"] = "High-risk response; verify before acting on it"
    
    # MEDIUM risk interventions
    elif risk_level == RiskLevel.MEDIUM.value:
        if halluc > 0.5:
            interventions.append(InterventionAction.WARN_USER)
            rationale["warning"] = f"Moderate hallucination risk ({halluc:.2%})"
        if citation_gap > 0.3:
            interventions.append(InterventionAction.FORCE_CITATIONS)
            rationale["citations"] = "Some claims may need citations"
        if uncertainty > 0.3:
            interventions.append(InterventionAction.ASK_CLARIFYING_QUESTION)
            rationale["clarification"] = "High uncertainty detected"
    
    # LOW risk: minimal interventions (just awareness)
    else:
        if uncertainty > 0.5:
            interventions.append(InterventionAction.WARN_USER)
            rationale["warning"] = "Answer contains significant uncertainty language"
    
    # Deduplicate
    interventions = list(set(interventions))
    
    return {
        "recommended_interventions": [a.value for a in interventions],
        "rationale": rationale,
        "severity": risk_level,
    }


def apply_interventions(interventions: List[str], response: str) -> Dict[str, Any]:
    """Apply interventions to a response. Return modified response + metadata."""
    result = {
        "original_response": response,
        "modified_response": response,
        "applied_interventions": [],
        "warnings": [],
        "instructions": [],
    }
    
    if InterventionAction.WARN_USER.value in interventions:
        warning = "⚠️ WARNING: This response has been flagged for potential hallucination. Verify critical facts before acting on this information."
        result["warnings"].append(warning)
    
    if InterventionAction.FORCE_CITATIONS.value in interventions:
        result["instructions"].append("Please provide citations or sources for your claims.")
    
    if InterventionAction.ACTIVATE_STRICT_GUARDRAIL.value in interventions:
        result["instructions"].append("Strict guardrail activated: Please limit claims to well-established facts only.")
    
    if InterventionAction.ASK_CLARIFYING_QUESTION.value in interventions:
        result["instructions"].append("Please clarify any contradictory statements in your response.")
    
    if InterventionAction.SWITCH_MODEL.value in interventions:
        result["instructions"].append("Consider re-running with a more reliable model (e.g., higher-parameter or specialized).")
    
    result["applied_interventions"] = interventions
    return result


__all__ = [
    "RiskLevel",
    "InterventionAction",
    "classify_risk_level",
    "recommend_interventions",
    "apply_interventions",
]
