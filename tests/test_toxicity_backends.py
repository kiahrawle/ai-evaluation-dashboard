import pytest

import src.evaluators.toxicity_backends as tb
from src.evaluators.toxicity_backends import get_backend, detoxify_backend


def test_lexical_backend_is_none():
    # None signals "use the built-in lexical fallback" (explicit opt-in).
    assert get_backend("lexical") is None
    assert get_backend(None) is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("perspective")


def test_get_backend_detoxify_fails_fast_when_missing(monkeypatch):
    # No silent fallback: requesting detoxify without it installed must raise.
    monkeypatch.setattr(tb, "detoxify_available", lambda: False)
    with pytest.raises(ImportError, match="CRITICAL"):
        get_backend("detoxify")


def test_get_backend_detoxify_lazy_when_available(monkeypatch):
    # When detoxify is present, get_backend returns a callable WITHOUT loading
    # the model (no download at resolve time).
    monkeypatch.setattr(tb, "detoxify_available", lambda: True)
    monkeypatch.setattr(tb, "_detox_model", None)
    backend = get_backend("detoxify")
    assert callable(backend)
    assert tb._detox_model is None  # still not loaded


def test_detoxify_backend_clear_error_when_missing(monkeypatch):
    # The lazy scorer itself also raises a clear error if the import fails.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "detoxify":
            raise ImportError("no detoxify")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(tb, "_detox_model", None)
    backend = detoxify_backend()
    with pytest.raises(RuntimeError, match="detoxify"):
        backend("some text")
