"""Real-time risk monitoring during generation (Step 11 - the live layer).

`scan_for_risk_markers` only runs once, on a finished answer. This module runs
the same token-level detectors *incrementally* as a model streams its answer, so
risk can be tracked, displayed, and acted on before generation even finishes.

    monitor = RealTimeRiskMonitor()
    for chunk in stream:          # whatever yields text deltas
        snap = monitor.update(chunk)
        print(snap["risk_level"], snap["risk_score"])

`stream_and_monitor` wires this to the streaming generator and the existing
intervention recommender (src/guardrails) so the live path ends in an action,
not just a number.
"""
from typing import Dict, Any, List, Iterable, Optional, Callable

from src.utils.risk_markers import scan_for_risk_markers

# Streaming risk is marker-based: mid-generation we have no references or
# retrieval, only the text so far. Weights lean on the strongest signals.
STREAM_WEIGHTS = {
    "risky": 0.35,          # "I can't verify", "I'm guessing"
    "citation_gap": 0.25,   # "studies show" with no citation
    "contradiction": 0.25,  # "actually", "wait", "let me revise"
    "uncertainty": 0.15,    # "I think", "probably"
}

# Match the platform's LOW/MEDIUM/HIGH bands (src/risk, src/guardrails).
LOW_MAX = 0.33
MEDIUM_MAX = 0.67


def _level(risk_score: float) -> str:
    if risk_score < LOW_MAX:
        return "LOW"
    if risk_score < MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def stream_risk_from_markers(markers: Dict[str, Any]) -> float:
    """Collapse a scan_for_risk_markers result into a single 0-1 risk score."""
    risk = (
        markers["risky_phrases"]["risky_score"] * STREAM_WEIGHTS["risky"]
        + markers["citation_gaps"]["citation_gap_score"] * STREAM_WEIGHTS["citation_gap"]
        + markers["contradictions"]["contradiction_score"] * STREAM_WEIGHTS["contradiction"]
        + markers["uncertainty"]["uncertainty_score"] * STREAM_WEIGHTS["uncertainty"]
    )
    return float(max(0.0, min(1.0, risk)))


class RealTimeRiskMonitor:
    """Accumulates streamed text and re-scores risk on every update."""

    def __init__(self, high_threshold: float = MEDIUM_MAX):
        self.text = ""
        self.high_threshold = high_threshold
        self.snapshots: List[Dict[str, Any]] = []
        self.first_high_at: Optional[int] = None

    def update(self, chunk: str) -> Dict[str, Any]:
        """Feed a text delta; return the current risk snapshot."""
        self.text += chunk or ""
        markers = scan_for_risk_markers(self.text)
        risk_score = stream_risk_from_markers(markers)
        level = _level(risk_score)
        snap = {
            "chars": len(self.text),
            "risk_score": risk_score,
            "risk_level": level,
            "uncertainty_score": markers["uncertainty"]["uncertainty_score"],
            "risky_score": markers["risky_phrases"]["risky_score"],
            "contradiction_score": markers["contradictions"]["contradiction_score"],
            "citation_gap_score": markers["citation_gaps"]["citation_gap_score"],
            "should_intervene": risk_score >= self.high_threshold,
        }
        if snap["should_intervene"] and self.first_high_at is None:
            self.first_high_at = len(self.text)
        self.snapshots.append(snap)
        return snap

    def snapshot(self) -> Dict[str, Any]:
        return self.snapshots[-1] if self.snapshots else {
            "chars": 0, "risk_score": 0.0, "risk_level": "LOW",
            "uncertainty_score": 0.0, "risky_score": 0.0,
            "contradiction_score": 0.0, "citation_gap_score": 0.0,
            "should_intervene": False,
        }

    def recommend(self) -> Dict[str, Any]:
        """Map the current risk state to concrete interventions (Step 13)."""
        from src import guardrails
        snap = self.snapshot()
        risk_info = {
            "risk_score": snap["risk_score"],
            "risk_level": snap["risk_level"],
            "hallucination_score": snap["risk_score"],
            "citation_supported": True,
            "uncertainty_score": snap["uncertainty_score"],
            "contradiction_score": snap["contradiction_score"],
            "citation_gap_score": snap["citation_gap_score"],
        }
        return guardrails.recommend_interventions(risk_info)


def stream_and_monitor(
    model: str,
    question: str,
    token_source: Optional[Iterable[str]] = None,
    on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    high_threshold: float = MEDIUM_MAX,
) -> Dict[str, Any]:
    """Stream an answer while tracking risk; end with recommended interventions.

    `token_source` defaults to the live streaming generator; pass an iterable of
    text chunks to drive it without an API call (tests, replay).
    """
    if token_source is None:
        from src import models
        token_source = models.stream_chat(model, question)

    monitor = RealTimeRiskMonitor(high_threshold=high_threshold)
    for chunk in token_source:
        snap = monitor.update(chunk)
        if on_update is not None:
            on_update(snap)

    return {
        "question": question,
        "model": model,
        "final_text": monitor.text,
        "final_risk": monitor.snapshot(),
        "risk_trace": monitor.snapshots,
        "first_high_at": monitor.first_high_at,
        "recommended": monitor.recommend(),
    }


__all__ = [
    "RealTimeRiskMonitor",
    "stream_and_monitor",
    "stream_risk_from_markers",
]
