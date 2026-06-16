"""Self-consistency (sampling-based) hallucination signal.

Reference-free intuition (SelfCheckGPT-style): when a model *knows* a fact, it
answers a question the same way across independent samples; when it is
hallucinating, the samples diverge. We draw several answers at a non-zero
temperature and measure how much they agree.

    consistency           = mean pairwise cosine similarity (rescaled to 0-1)
    hallucination_signal  = 1 - consistency

`measure_consistency` is pure (operates on a list of strings) so it is cheap to
test and reuse; `self_consistency_score` wires it to the sampling generator.
"""
from typing import Dict, Any, List, Callable, Optional
from itertools import combinations
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


def measure_consistency(responses: List[str]) -> Dict[str, Any]:
    """Measure agreement across sampled responses.

    With fewer than two responses there is nothing to disagree on, so we report
    full consistency (no hallucination signal).
    """
    clean = [r for r in (responses or []) if r and r.strip()]
    n = len(clean)
    if n < 2:
        return {
            "n": n,
            "mean_pairwise_similarity": 1.0,
            "consistency": 1.0,
            "hallucination_signal": 0.0,
        }

    m = _embed_model()
    emb = m.encode(clean, convert_to_tensor=True)
    sims = [util.cos_sim(emb[i], emb[j]).item() for i, j in combinations(range(n), 2)]
    mean_sim = sum(sims) / len(sims)
    # Cosine similarity is in [-1, 1]; rescale to [0, 1] for an intuitive score.
    consistency = max(0.0, min(1.0, (mean_sim + 1.0) / 2.0))
    return {
        "n": n,
        "mean_pairwise_similarity": float(mean_sim),
        "consistency": float(consistency),
        "hallucination_signal": float(1.0 - consistency),
    }


def self_consistency_score(
    model: str,
    question: str,
    n: int = 5,
    temperature: float = 0.7,
    sampler: Optional[Callable[[str, str, int, float], List[str]]] = None,
) -> Dict[str, Any]:
    """Sample `n` answers for `question` and measure their agreement.

    `sampler` defaults to the model generator's `sample`; inject a stub in tests
    to avoid API calls.
    """
    if sampler is None:
        from src import models
        sampler = models.sample
    samples = sampler(model, question, n, temperature)
    result = measure_consistency(samples)
    result["samples"] = samples
    return result


__all__ = ["measure_consistency", "self_consistency_score"]
