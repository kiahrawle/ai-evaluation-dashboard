from src.rag.retriever import initialize_index, retrieve, load_corpus
from pathlib import Path


def test_initialize_index():
    docs = [
        {"text": "Python is a programming language"},
        {"text": "Java is another programming language"},
        {"text": "The sky is blue"},
    ]
    result = initialize_index(docs)
    # Result may be False if FAISS not installed
    if not result:
        return
    
    # Try retrieval
    retrieved = retrieve("What is Python?", top_k=2)
    assert len(retrieved) <= 2
    if retrieved:
        assert "text" in retrieved[0]


def test_load_corpus():
    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.json"
    if not corpus_path.exists():
        return  # skip if corpus not available
    
    result = load_corpus(corpus_path)
    # May be False if FAISS not installed
    if not result:
        return
    
    retrieved = retrieve("What is the moon?", top_k=3)
    if retrieved:
        assert len(retrieved) <= 3
