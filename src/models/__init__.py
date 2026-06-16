"""Generation layer for the models under test.

`generate_all`/`chat` are the provider-specific entry points; swap the
implementation in `generator.py` to evaluate a different provider.
"""
from .generator import chat, generate_all, sample, stream_chat

__all__ = ["chat", "generate_all", "sample", "stream_chat"]
