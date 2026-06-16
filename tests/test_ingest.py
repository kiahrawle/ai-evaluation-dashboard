import pytest

from src.rag.ingest import extract_text, chunk_text, ingest_document


def test_extract_text_utf8():
    assert extract_text("doc.txt", "hello world".encode("utf-8")) == "hello world"


def test_extract_text_latin1_fallback():
    # 0xe9 is 'é' in latin-1 and invalid as standalone utf-8.
    out = extract_text("doc.txt", b"caf\xe9")
    assert "caf" in out


def test_chunk_text_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) == 2
    assert chunks[0]["text"].startswith("First")
    assert all("text" in c for c in chunks)


def test_chunk_text_splits_long_paragraph():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, max_chars=120)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 200 for c in chunks)  # roughly bounded


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_ingest_document_combines():
    docs = ingest_document("notes.md", b"Alpha fact.\n\nBeta fact.")
    assert len(docs) == 2


def test_extract_pdf_without_dep_raises(monkeypatch):
    # Simulate pypdf missing -> clear error, not a crash.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pypdf"):
        extract_text("doc.pdf", b"%PDF-1.4 ...")
