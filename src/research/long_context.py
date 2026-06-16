"""Long-context evaluation - track memory drift, contradictions, and context loss."""
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import re

_embedder = None


def _embed_model():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def detect_memory_drift(statements: List[str]) -> Dict[str, Any]:
    """Detect if later statements contradict earlier ones (memory drift).
    
    Input: List of sequential statements from the same conversation
    Returns: {
        "has_drift": bool,
        "drift_score": float (0-1),
        "contradictions": List[Dict],
        "first_contradictions": List[Tuple[int, int]]
    }
    """
    if len(statements) < 2:
        return {
            "has_drift": False,
            "drift_score": 0.0,
            "contradictions": [],
            "first_contradictions": [],
        }
    
    m = _embed_model()
    embeddings = m.encode(statements, convert_to_tensor=True)
    
    contradictions = []
    
    # Compare each statement to all earlier ones
    for i in range(1, len(statements)):
        for j in range(i):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            # Low similarity to an earlier statement might indicate drift
            if sim < 0.3:  # Potential contradiction threshold
                contradictions.append({
                    "statement_a_idx": j,
                    "statement_b_idx": i,
                    "statement_a": statements[j],
                    "statement_b": statements[i],
                    "similarity": float(sim),
                })
    
    has_drift = len(contradictions) > 0
    # Calculate drift score: higher = more contradictions
    drift_score = min(1.0, len(contradictions) / len(statements))
    
    # Track first contradiction pair
    first_contradictions = [(c["statement_a_idx"], c["statement_b_idx"]) for c in contradictions[:3]]
    
    return {
        "has_drift": has_drift,
        "drift_score": float(drift_score),
        "contradictions": contradictions,
        "first_contradictions": first_contradictions,
    }


def detect_context_loss(context: str, later_statement: str) -> float:
    """Measure if later statement fails to reference earlier context.
    
    Returns: 0-1 score indicating context loss (higher = more loss)
    """
    m = _embed_model()
    context_emb = m.encode(context, convert_to_tensor=True)
    statement_emb = m.encode(later_statement, convert_to_tensor=True)
    
    similarity = util.cos_sim(context_emb, statement_emb).item()
    # Invert: low similarity = high context loss
    context_loss = 1.0 - ((similarity + 1.0) / 2.0)
    return float(max(0.0, min(1.0, context_loss)))


def track_entity_consistency(statements: List[str]) -> Dict[str, Any]:
    """Track if entities (names, facts) are mentioned consistently across statements.
    
    Returns: {
        "entities": Dict[str, List[int]],  # entity -> statement indices where mentioned
        "consistency_score": float,
        "inconsistent_entities": List[str]
    }
    """
    # Simple entity extraction (names, numbers, capitalized words)
    entities = {}
    
    for idx, stmt in enumerate(statements):
        # Find capitalized words (rough entity detection)
        words = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", stmt)
        for word in words:
            if word not in entities:
                entities[word] = []
            entities[word].append(idx)
    
    # Score: entities mentioned consistently in order
    consistency_score = 0.0
    inconsistent = []
    
    if entities:
        for entity, indices in entities.items():
            # Check if mentions are spread out (not clumped)
            if len(indices) > 1:
                # If entity is mentioned in consecutive or close statements, it's consistent
                diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                avg_gap = sum(diffs) / len(diffs) if diffs else 0
                if avg_gap > len(statements) / 2:
                    inconsistent.append(entity)
        
        consistency_score = 1.0 - (len(inconsistent) / len(entities)) if entities else 1.0
    
    return {
        "entities": entities,
        "consistency_score": float(max(0.0, min(1.0, consistency_score))),
        "inconsistent_entities": inconsistent,
    }


def evaluate_long_context(statements: List[str]) -> Dict[str, Any]:
    """Comprehensive long-context evaluation.
    
    Input: List of statements/responses in sequence
    Returns: {
        "memory_drift": Dict,
        "entity_consistency": Dict,
        "context_preservation": float,
        "overall_long_context_score": float
    }
    """
    memory_drift = detect_memory_drift(statements)
    entity_consistency = track_entity_consistency(statements)
    
    # Context preservation: average how well each statement references previous context
    context_loss_scores = []
    for i in range(1, len(statements)):
        context = " ".join(statements[:i])
        loss = detect_context_loss(context, statements[i])
        context_loss_scores.append(loss)
    
    avg_context_loss = sum(context_loss_scores) / len(context_loss_scores) if context_loss_scores else 0.0
    context_preservation = 1.0 - avg_context_loss
    
    # Overall score: average of all factors
    overall_score = (
        (1.0 - memory_drift["drift_score"]) * 0.4 +
        entity_consistency["consistency_score"] * 0.3 +
        context_preservation * 0.3
    )
    
    return {
        "memory_drift": memory_drift,
        "entity_consistency": entity_consistency,
        "context_preservation": float(context_preservation),
        "overall_long_context_score": float(overall_score),
        "assessment": (
            "Excellent" if overall_score > 0.8 else
            "Good" if overall_score > 0.6 else
            "Fair" if overall_score > 0.4 else
            "Poor"
        ),
    }


__all__ = [
    "detect_memory_drift",
    "detect_context_loss",
    "track_entity_consistency",
    "evaluate_long_context",
]
