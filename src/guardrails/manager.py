from typing import Dict


TRIGGER_HALLUCINATION_THRESHOLD = 0.6
TRIGGER_RISK_THRESHOLD = 0.5


def should_trigger(er: Dict) -> bool:
    hs = er.get("hallucination_score") or 0.0
    rs = er.get("risk_score") or 0.0
    return hs >= TRIGGER_HALLUCINATION_THRESHOLD or rs >= TRIGGER_RISK_THRESHOLD


def enforcement_prompt() -> str:
    return (
        "Be conservative and factual. If unsure, say you are unsure. "
        "Do not fabricate facts or details. When a question rests on a "
        "common misconception, explicitly flag the uncertainty."
    )


__all__ = ["should_trigger", "enforcement_prompt"]
