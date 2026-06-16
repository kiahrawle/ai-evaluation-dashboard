"""Topic detection using embeddings and keyword-based classification."""
from typing import Dict, Any, Tuple
import re
from sentence_transformers import SentenceTransformer, util

_embedder = None

# Topic keywords for keyword-based fallback
TOPIC_KEYWORDS = {
    "medical": [
        "disease", "symptom", "treatment", "medication", "diagnosis",
        "patient", "doctor", "hospital", "health", "vaccine", "cancer",
        "heart", "brain", "surgery", "clinical", "dose"
    ],
    "finance": [
        "investment", "stock", "bond", "portfolio", "trading", "profit",
        "loss", "earnings", "dividend", "loan", "credit", "debt", "tax",
        "market", "exchange", "price", "revenue", "budget"
    ],
    "legal": [
        "law", "court", "judge", "attorney", "lawsuit", "contract",
        "agreement", "liability", "rights", "copyright", "patent",
        "copyright", "constitution", "regulation", "statute", "verdict"
    ],
    "science": [
        "experiment", "hypothesis", "theory", "physics", "chemistry",
        "biology", "research", "data", "analysis", "evidence",
        "quantum", "molecule", "atom", "energy", "force"
    ],
    "politics": [
        "election", "vote", "politician", "government", "law",
        "policy", "party", "campaign", "congress", "senate",
        "president", "parliament", "political", "parliament"
    ],
}

TOPIC_REFERENCE_TEXTS = {
    "medical": "This question is about medical conditions, treatments, medications, or health topics.",
    "finance": "This question is about financial markets, investments, banking, or economic matters.",
    "legal": "This question is about laws, courts, legal rights, or regulatory compliance.",
    "science": "This question is about scientific research, physics, chemistry, biology, or experimental data.",
    "politics": "This question is about elections, government policy, politicians, or political systems.",
}


def _embed_model():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def detect_topic_keyword_based(text: str) -> Tuple[str, float]:
    """Detect topic using keyword matching (fast, no embeddings)."""
    text_lower = text.lower()
    topic_scores = {}
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        topic_scores[topic] = count
    
    if not topic_scores or max(topic_scores.values()) == 0:
        return "general", 0.0
    
    best_topic = max(topic_scores.items(), key=lambda x: x[1])
    score = min(1.0, best_topic[1] / 5.0)  # Normalize by typical keyword frequency
    return best_topic[0], float(score)


def detect_topic_embedding_based(text: str) -> Tuple[str, float]:
    """Detect topic using semantic similarity (more accurate but slower)."""
    try:
        m = _embed_model()
        text_emb = m.encode(text, convert_to_tensor=True)
        
        topic_sims = {}
        for topic, ref_text in TOPIC_REFERENCE_TEXTS.items():
            ref_emb = m.encode(ref_text, convert_to_tensor=True)
            sim = util.cos_sim(text_emb, ref_emb).item()
            topic_sims[topic] = sim
        
        if not topic_sims:
            return "general", 0.0
        
        best_topic, best_score = max(topic_sims.items(), key=lambda x: x[1])
        score = float(max(0.0, min(1.0, (best_score + 1.0) / 2.0)))
        
        # Only return topic if confidence is reasonable
        if score < 0.4:
            return "general", score
        
        return best_topic, score
    except Exception:
        return "general", 0.0


def detect_topic(text: str, method: str = "hybrid") -> Dict[str, Any]:
    """Detect topic with confidence.
    
    Methods:
    - "keyword": Fast keyword-based detection
    - "embedding": Semantic similarity (slower, more accurate)
    - "hybrid": Try embedding first, fallback to keyword
    
    Returns: {
        "topic": str,
        "confidence": float,
        "method": str
    }
    """
    if method == "keyword":
        topic, conf = detect_topic_keyword_based(text)
        return {"topic": topic, "confidence": conf, "method": "keyword"}
    
    elif method == "embedding":
        topic, conf = detect_topic_embedding_based(text)
        return {"topic": topic, "confidence": conf, "method": "embedding"}
    
    elif method == "hybrid":
        # Try embedding first (more accurate)
        try:
            topic, conf = detect_topic_embedding_based(text)
            if conf >= 0.5:
                return {"topic": topic, "confidence": conf, "method": "embedding"}
        except Exception:
            pass
        # Fall back to keyword
        topic, conf = detect_topic_keyword_based(text)
        return {"topic": topic, "confidence": conf, "method": "keyword"}
    
    return {"topic": "general", "confidence": 0.0, "method": "unknown"}


__all__ = ["detect_topic", "detect_topic_keyword_based", "detect_topic_embedding_based"]
