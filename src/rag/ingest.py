"""Document ingestion for the upload -> RAG flow.

Turns an uploaded file into a list of {"text": ...} chunks ready for
`rag.initialize_index`. Kept Streamlit-free so it can be unit-tested.
"""
from typing import List, Dict
import re


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from uploaded bytes based on file extension.

    Supports .txt / .md natively and .pdf when `pypdf` is installed; raises a
    clear error otherwise.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf(data)
    # txt / md / anything text-like
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "PDF upload needs the `pypdf` package. Install it with "
            "`pip install pypdf`, or upload a .txt/.md file instead."
        ) from e
    import io
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text: str, max_chars: int = 500, overlap: int = 0) -> List[Dict[str, str]]:
    """Split text into passage chunks of roughly `max_chars`, on sentence/para
    boundaries where possible. Returns [{"text": ...}, ...] for the retriever.
    """
    if not text or not text.strip():
        return []
    # Prefer paragraph boundaries, then pack sentences up to max_chars.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = f"{buf} {s}".strip()
                else:
                    if buf:
                        units.append(buf)
                    buf = s
            if buf:
                units.append(buf)
    return [{"text": u} for u in units if u.strip()]


def ingest_document(filename: str, data: bytes, max_chars: int = 500) -> List[Dict[str, str]]:
    """Convenience: extract + chunk in one call."""
    return chunk_text(extract_text(filename, data), max_chars=max_chars)


__all__ = ["extract_text", "chunk_text", "ingest_document"]
