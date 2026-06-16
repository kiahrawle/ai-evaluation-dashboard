"""Core data schema for evaluation objects."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Universal evaluation object used across the platform.

    Fields are intentionally permissive (Optional) so components can be
    incrementally populated.
    """
    prompt: str
    response: str
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)

    truthful: Optional[bool] = None
    informative: Optional[bool] = None
    grounded: Optional[bool] = None

    hallucination_score: Optional[float] = None
    risk_score: Optional[float] = None
    confidence_score: Optional[float] = None

    guardrail_triggered: bool = False
    failure_type: Optional[str] = None

    # Optional contextual fields
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    def to_json(self, **kwargs) -> str:
        return self.json(**kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationResult":
        return cls(**d)


__all__ = ["EvaluationResult"]
