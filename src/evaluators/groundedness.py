"""Groundedness and citation verification.

Score how well an answer is supported by retrieved documents and verify citations.
"""
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

_embedder = None


def _embed_model():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def score_groundedness(retrieved_docs: List[Dict], answer: str) -> float:
    """Score how much the answer is supported by retrieved docs (0-1)."""
    if not retrieved_docs:
        return 0.0
    texts = [d.get("text", "") for d in retrieved_docs if d.get("text")]
    if not texts:
        return 0.0
    m = _embed_model()
    ans = m.encode(answer, convert_to_tensor=True)
    docs_emb = m.encode(texts, convert_to_tensor=True)
    best = util.cos_sim(ans, docs_emb).max().item()
    score = max(0.0, min(1.0, (best + 1.0) / 2.0))
    return float(score)


def extract_claims(answer: str) -> List[str]:
    """Extract major factual claims from the answer (simple heuristic)."""
    sentences = re.split(r'[.!?]+', answer)
    claims = [s.strip() for s in sentences if len(s.strip()) > 10]
    return claims[:5]  # limit to top 5 claims


SUPPORT_SIM_THRESHOLD = 0.5     # claim is "supported" if its best doc sim exceeds this


def verify_citation_support(answer: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
    """Check if claims in answer are supported by retrieved docs, and LINK each
    claim to the specific document that best supports it (evidence extraction).

    Returns: {
        "citation_supported": bool,
        "supported_claims": int,
        "total_claims": int,
        "support_score": float,
        "unsupported_claims": List[str],
        "evidence": [ {"claim", "supported", "similarity",
                       "doc_index", "evidence_text"}, ... ],
    }
    """
    if not retrieved_docs:
        return {
            "citation_supported": False,
            "supported_claims": 0,
            "total_claims": 0,
            "support_score": 0.0,
            "unsupported_claims": [],
            "evidence": [],
        }

    claims = extract_claims(answer)
    if not claims:
        return {
            "citation_supported": True,
            "supported_claims": 0,
            "total_claims": 0,
            "support_score": 1.0,
            "unsupported_claims": [],
            "evidence": [],
        }

    texts = [d.get("text", "") for d in retrieved_docs]
    m = _embed_model()
    # Encode each document ONCE so we can attribute support to a specific doc
    # (citation linking) instead of one concatenated blob.
    doc_embs = m.encode(texts, convert_to_tensor=True)

    supported = 0
    unsupported: List[str] = []
    evidence: List[Dict[str, Any]] = []

    for claim in claims:
        claim_emb = m.encode(claim, convert_to_tensor=True)
        sims = util.cos_sim(claim_emb, doc_embs)[0]
        best_idx = int(sims.argmax().item())
        best_sim = float(sims[best_idx].item())
        is_supported = best_sim > SUPPORT_SIM_THRESHOLD
        if is_supported:
            supported += 1
        else:
            unsupported.append(claim)
        evidence.append({
            "claim": claim,
            "supported": is_supported,
            "similarity": best_sim,
            "doc_index": best_idx if is_supported else None,
            "evidence_text": texts[best_idx] if is_supported else None,
        })

    support_score = supported / len(claims) if claims else 1.0
    citation_supported = support_score >= 0.6  # majority support threshold

    return {
        "citation_supported": citation_supported,
        "supported_claims": supported,
        "total_claims": len(claims),
        "support_score": float(support_score),
        "unsupported_claims": unsupported,
        "evidence": evidence,
    }


# --- Numeric contradiction detection (Revenue=12M in docs vs LLM says 50M) ----
# Pure embedding similarity treats "Revenue = 50M" and "Revenue = 12M" as nearly
# identical (same topic) and would call the answer "supported". That is exactly
# the failure mode that matters for grounded factual claims, so we additionally
# compare the *numbers* attached to shared context words.

_SCALES = {
    "trillion": 1e12, "tn": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}
_NUM_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "of", "to", "in", "at", "on",
    "our", "its", "their", "approximately", "about", "around", "roughly", "and",
    "or", "for", "with", "by", "than", "over", "under", "nearly", "almost",
}
_NUM_RE = re.compile(
    r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(trillion|billion|million|thousand|tn|bn|mn|[bmk])?\s*(%|percent)?",
    re.IGNORECASE,
)


def _extract_number_facts(text: str) -> List[Dict[str, Any]]:
    """Pull (value, unit, context-words) for every number mentioned in `text`."""
    facts: List[Dict[str, Any]] = []
    for mobj in _NUM_RE.finditer(text):
        raw = mobj.group(1)
        if not raw:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        scale_word = (mobj.group(2) or "").lower()
        if scale_word in _SCALES:
            value *= _SCALES[scale_word]
        unit = "percent" if mobj.group(3) else "number"
        # Context = salient words in the ~6 chars... actually ~5 words before.
        prefix = text[: mobj.start()].lower()
        words = re.findall(r"[a-z]+", prefix)[-5:]
        context = {w for w in words if w not in _NUM_STOPWORDS and len(w) > 2}
        facts.append({"value": value, "unit": unit, "context": context, "raw": mobj.group(0).strip()})
    return facts


def detect_numeric_contradictions(
    answer: str, retrieved_docs: List[Dict], rel_tol: float = 0.1
) -> Dict[str, Any]:
    """Flag numbers in `answer` that contradict numbers in the retrieved docs.

    Two numbers contradict when they share a context word (e.g. "revenue") and
    the same unit, but differ by more than `rel_tol` (relative). "27 days" vs
    "27.3 days" (0.011) is NOT a contradiction; "50M" vs "12M" is.
    """
    answer_facts = _extract_number_facts(answer)
    doc_text = " ".join(d.get("text", "") for d in retrieved_docs)
    doc_facts = _extract_number_facts(doc_text)

    contradictions: List[Dict[str, Any]] = []
    for af in answer_facts:
        for df in doc_facts:
            if af["unit"] != df["unit"]:
                continue
            if not (af["context"] & df["context"]):
                continue
            denom = max(abs(af["value"]), abs(df["value"]), 1e-9)
            if abs(af["value"] - df["value"]) / denom > rel_tol:
                contradictions.append({
                    "context": sorted(af["context"] & df["context"]),
                    "answer_value": af["raw"],
                    "doc_value": df["raw"],
                })
                break  # one contradiction per answer number is enough

    return {
        "contradiction_detected": bool(contradictions),
        "num_contradictions": len(contradictions),
        "contradictions": contradictions,
    }


__all__ = [
    "score_groundedness", "extract_claims", "verify_citation_support",
    "detect_numeric_contradictions",
]
