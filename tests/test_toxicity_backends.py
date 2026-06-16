import pytest

from src.evaluators.toxicity_backends import get_backend, detoxify_backend


def test_lexical_backend_is_none():
    # None signals "use the built-in lexical fallback".
    assert get_backend("lexical") is None
    assert get_backend(None) is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("perspective")


def test_detoxify_backend_is_lazy():
    # Constructing the backend must NOT import detoxify or download a model.
    backend = get_backend("detoxify")
    assert callable(backend)


def test_detoxify_backend_clear_error_when_missing(monkeypatch):
    # Simulate detoxify not installed -> calling the scorer raises a clear error.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "detoxify":
            raise ImportError("no detoxify")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Reset any cached model so the import path is exercised.
    import src.evaluators.toxicity_backends as tb
    monkeypatch.setattr(tb, "_detox_model", None)
    backend = detoxify_backend()
    with pytest.raises(RuntimeError, match="detoxify"):
        backend("some text")
