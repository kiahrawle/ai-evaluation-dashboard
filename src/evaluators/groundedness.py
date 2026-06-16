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
# Old approach: gate on shared *context words* before comparing numbers. That is
# lexically blind — "90% of patients" vs "45% of patients" describes the same
# quantity with different surrounding words ("symptoms" vs "improvement") and was
# missed. New approach: embed the sentences that contain numbers and gate on
# semantic similarity, so paraphrases of the same fact are still compared.

_SCALES = {
    "trillion": 1e12, "tn": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}
_NUM_RE = re.compile(
    r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(trillion|billion|million|thousand|tn|bn|mn|[bmk])?\s*(%|percent)?",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Tuned for all-MiniLM-L6-v2: short factual sentences about the same quantity
# score ~0.5-0.75, while different topics score ~0.2-0.35, so 0.45 separates
# them cleanly. (A naive 0.75 would miss most real contradictions on this model.)
SEM_SIM_THRESHOLD = 0.45


def _extract_number_facts(text: str) -> List[Dict[str, Any]]:
    """Pull (value, unit, raw) for every number mentioned in `text`."""
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
        facts.append({"value": value, "unit": unit, "raw": mobj.group(0).strip()})
    return facts


def _sentences_with_numbers(text: str) -> List[Dict[str, Any]]:
    out = []
    for sent in _SENT_SPLIT.split(text or ""):
        sent = sent.strip()
        facts = _extract_number_facts(sent)
        if facts:
            out.append({"sentence": sent, "facts": facts})
    return out


def _number_conflict(a_facts, d_facts, rel_tol):
    """Return (answer_raw, doc_raw) if an answer number has no matching doc number
    of the same unit (within rel_tol), else None."""
    by_unit: Dict[str, List[Dict[str, Any]]] = {}
    for f in d_facts:
        by_unit.setdefault(f["unit"], []).append(f)
    for af in a_facts:
        candidates = by_unit.get(af["unit"])
        if not candidates:
            continue
        matched = any(
            abs(af["value"] - df["value"]) / max(abs(af["value"]), abs(df["value"]), 1e-9) <= rel_tol
            for df in candidates
        )
        if not matched:
            closest = min(candidates, key=lambda df: abs(af["value"] - df["value"]))
            return af["raw"], closest["raw"]
    return None


def detect_numeric_contradictions(
    answer: str, retrieved_docs: List[Dict],
    rel_tol: float = 0.1, sim_threshold: float = SEM_SIM_THRESHOLD,
) -> Dict[str, Any]:
    """Flag numbers in `answer` that contradict numbers in the retrieved docs.

    For each answer sentence containing a number, find the most semantically
    similar doc sentence (embeddings). If that similarity clears `sim_threshold`
    (the sentences are about the same thing) but a number of the same unit
    differs by more than `rel_tol`, it's a contradiction. "27 days" vs "27.3
    days" (within tol) is not; "50M" vs "12M" revenue, or "90%" vs "45%" of
    patients, is — even when the wording differs.
    """
    empty = {"contradiction_detected": False, "num_contradictions": 0, "contradictions": []}
    answer_sents = _sentences_with_numbers(answer)
    doc_text = " ".join(d.get("text", "") for d in retrieved_docs)
    doc_sents = _sentences_with_numbers(doc_text)
    if not answer_sents or not doc_sents:
        return empty

    m = _embed_model()
    a_emb = m.encode([s["sentence"] for s in answer_sents], convert_to_tensor=True)
    d_emb = m.encode([s["sentence"] for s in doc_sents], convert_to_tensor=True)
    sims = util.cos_sim(a_emb, d_emb)

    contradictions: List[Dict[str, Any]] = []
    for i, a in enumerate(answer_sents):
        j = int(sims[i].argmax().item())
        best = float(sims[i][j].item())
        if best < sim_threshold:
            continue
        conflict = _number_conflict(a["facts"], doc_sents[j]["facts"], rel_tol)
        if conflict:
            contradictions.append({
                "context": doc_sents[j]["sentence"],
                "answer_value": conflict[0],
                "doc_value": conflict[1],
                "similarity": round(best, 3),
            })

    return {
        "contradiction_detected": bool(contradictions),
        "num_contradictions": len(contradictions),
        "contradictions": contradictions,
    }


__all__ = [
    "score_groundedness", "extract_claims", "verify_citation_support",
    "detect_numeric_contradictions",
]
