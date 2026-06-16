"""Toxicity scoring backends.

`tasks.score_toxicity`'s built-in lexical scorer is a placeholder. `detoxify`
(unitary/detoxify) is a real trained toxicity classifier: once its weights
download (once, on first use) it runs locally on CPU. Selecting the detoxify
backend turns the toxicity benchmark from a placeholder into a real metric.

Backends are `str -> float` callables (0-1 toxicity). `get_backend("lexical")`
returns None, which tells the scorer/runner to use the built-in lexical fallback.
The detoxify model is loaded lazily (first call), so importing this module and
constructing the backend never downloads anything.
"""
from typing import Callable, Optional
import importlib.util

_detox_model = None


def detoxify_available() -> bool:
    """True if the `detoxify` package is importable (no model load)."""
    return importlib.util.find_spec("detoxify") is not None


def detoxify_backend(model_name: str = "original") -> Callable[[str], float]:
    """Return a lazy detoxify-backed scorer. Raises a clear error on first call
    if `detoxify` isn't installed."""
    def score(text: str) -> float:
        global _detox_model
        if _detox_model is None:
            try:
                from detoxify import Detoxify
            except ImportError as e:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "Toxicity backend 'detoxify' is not installed. "
                    "Run `pip install detoxify` (downloads model weights on first use)."
                ) from e
            _detox_model = Detoxify(model_name)
        out = _detox_model.predict(text or "")
        return float(out.get("toxicity", 0.0))
    return score


def get_backend(name: Optional[str]) -> Optional[Callable[[str], float]]:
    """Resolve a backend name to a callable (or None for the lexical fallback).

    FAIL FAST: requesting 'detoxify' when it isn't installed raises ImportError
    immediately, rather than silently degrading to the lexical scanner and
    reporting misleadingly low ("clean") toxicity during a formal benchmark. The
    lexical scorer is only used when the caller explicitly asks for it.
    """
    name = (name or "lexical").lower()
    if name in ("lexical", "none"):
        return None
    if name == "detoxify":
        if not detoxify_available():
            raise ImportError(
                "CRITICAL: the 'detoxify' toxicity classifier is not installed. "
                "Toxicity would otherwise fall back to a weak lexical scanner and "
                "report misleadingly 'clean' scores.\n"
                "Fix: `pip install detoxify`  —  or, if you knowingly accept "
                "unvalidated fallback metrics, pass `--toxicity-backend lexical`."
            )
        return detoxify_backend()
    raise ValueError(f"Unknown toxicity backend '{name}' (use 'detoxify' or 'lexical').")


__all__ = ["detoxify_backend", "get_backend", "detoxify_available"]
