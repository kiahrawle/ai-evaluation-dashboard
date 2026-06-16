"""FAISS-based vector retrieval for groundedness checks.

Builds an in-memory FAISS index from a corpus and retrieves top-k docs.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer

_index = None
_docs = None
_embedder = None


def _embed_model():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def initialize_index(documents: List[Dict[str, Any]]) -> bool:
    """Build FAISS index from a list of docs. Each doc should have 'text' key."""
    global _index, _docs
    if not HAS_FAISS:
        print("Warning: FAISS not installed. Retrieval will be disabled.")
        return False
    if not documents:
        return False
    _docs = documents
    texts = [d.get("text", "") for d in documents]
    m = _embed_model()
    embeddings = m.encode(texts, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings)
    return True


def retrieve(question: str, top_k: int = 5) -> List[Dict]:
    """Retrieve top-k docs similar to the question from FAISS index."""
    if not HAS_FAISS or _index is None or _docs is None:
        return []
    m = _embed_model()
    q_emb = m.encode([question], convert_to_tensor=False)
    q_emb = np.array(q_emb).astype("float32")
    distances, indices = _index.search(q_emb, min(top_k, len(_docs)))
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_docs):
            results.append(_docs[int(idx)])
    return results


def load_corpus(corpus_path: str | Path) -> bool:
    """Load a JSON corpus (list of {text: ...} dicts) and build index."""
    try:
        p = Path(corpus_path)
        with p.open() as f:
            docs = json.load(f)
        return initialize_index(docs)
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return False


__all__ = ["retrieve", "initialize_index", "load_corpus"]
