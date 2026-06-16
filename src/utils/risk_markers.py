"""Risk markers and token-level monitoring for detecting hallucination patterns."""
import re
from typing import Dict, List, Any

# Phrases indicating uncertainty or low confidence
UNCERTAINTY_PHRASES = [
    "i think",
    "i believe",
    "i'm not sure",
    "probably",
    "possibly",
    "might",
    "could",
    "may",
    "seems",
    "appears",
    "allegedly",
    "reportedly",
    "supposedly",
    "i don't know",
    "unclear",
]

# Phrases associated with hallucination risk
RISKY_PHRASES = [
    "as far as i know",
    "i don't have access to",
    "i can't verify",
    "i'm not certain",
    "this is my understanding",
    "based on my training",
    "i may be wrong",
    "i'm making an educated guess",
]

# Phrases indicating contradiction or revision
CONTRADICTION_MARKERS = [
    "actually",
    "wait",
    "correction",
    "i meant",
    "let me revise",
    "on second thought",
    "however",
    "contradicts",
    "contrary to",
]

# Phrases requiring citations
CITATION_REQUIRED_PHRASES = [
    "according to",
    "study shows",
    "research indicates",
    "statistics show",
    "data reveals",
    "evidence suggests",
]


def detect_uncertainty(text: str) -> Dict[str, Any]:
    """Detect uncertainty markers in text.
    
    Returns: {
        "uncertainty_score": float (0-1),
        "uncertainty_phrases": List[str],
        "count": int
    }
    """
    text_lower = text.lower()
    found = []
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in text_lower:
            found.append(phrase)
    score = min(1.0, len(found) * 0.2)
    return {
        "uncertainty_score": float(score),
        "uncertainty_phrases": found,
        "count": len(found),
    }


def detect_risky_phrases(text: str) -> Dict[str, Any]:
    """Detect risky/hallucination-prone phrases.
    
    Returns: {
        "risky_score": float (0-1),
        "risky_phrases": List[str],
        "count": int
    }
    """
    text_lower = text.lower()
    found = []
    for phrase in RISKY_PHRASES:
        if phrase in text_lower:
            found.append(phrase)
    score = min(1.0, len(found) * 0.25)
    return {
        "risky_score": float(score),
        "risky_phrases": found,
        "count": len(found),
    }


def detect_contradictions(text: str) -> Dict[str, Any]:
    """Detect contradiction markers suggesting revision or uncertainty.
    
    Returns: {
        "contradiction_score": float (0-1),
        "contradiction_markers": List[str],
        "count": int
    }
    """
    text_lower = text.lower()
    found = []
    for marker in CONTRADICTION_MARKERS:
        if marker in text_lower:
            found.append(marker)
    # Contradictions are somewhat common in natural speech
    score = min(0.7, len(found) * 0.15)
    return {
        "contradiction_score": float(score),
        "contradiction_markers": found,
        "count": len(found),
    }


def detect_citation_gaps(text: str) -> Dict[str, Any]:
    """Detect claims that require citations but may lack them.
    
    Returns: {
        "citation_gap_score": float (0-1),
        "has_claims_needing_citations": bool,
        "citation_phrases_found": List[str]
    }
    """
    text_lower = text.lower()
    found = []
    for phrase in CITATION_REQUIRED_PHRASES:
        if phrase in text_lower:
            found.append(phrase)
    
    # If claims are made that suggest citations but none provided, flag it
    has_url = "http" in text_lower or "www." in text_lower
    has_brackets = "[" in text and "]" in text
    has_citations = has_url or has_brackets
    
    gap_score = 0.0
    if found and not has_citations:
        gap_score = min(1.0, len(found) * 0.2)
    
    return {
        "citation_gap_score": float(gap_score),
        "has_claims_needing_citations": len(found) > 0,
        "citation_phrases_found": found,
    }


def scan_for_risk_markers(text: str) -> Dict[str, Any]:
    """Comprehensive risk marker scan. Aggregate all token-level risks."""
    uncertainty = detect_uncertainty(text)
    risky = detect_risky_phrases(text)
    contradictions = detect_contradictions(text)
    citation_gaps = detect_citation_gaps(text)
    
    return {
        "uncertainty": uncertainty,
        "risky_phrases": risky,
        "contradictions": contradictions,
        "citation_gaps": citation_gaps,
    }


__all__ = [
    "detect_uncertainty",
    "detect_risky_phrases",
    "detect_contradictions",
    "detect_citation_gaps",
    "scan_for_risk_markers",
]
