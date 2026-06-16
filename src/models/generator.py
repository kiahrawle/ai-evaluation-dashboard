"""Generation layer for the models UNDER TEST (Anthropic / Claude version).

Only `chat()` knows about Anthropic. Everything is cached to disk keyed by
(model, question) so re-runs cost nothing.
"""
import json
import hashlib
from pathlib import Path
from typing import Iterator

import anthropic
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

load_dotenv()                       # read .env BEFORE creating the client
_client = anthropic.Anthropic()     # reads ANTHROPIC_API_KEY from your .env


def _cache_path(model: str) -> Path:
    safe = model.replace("/", "_")
    return config.CACHE_DIR / f"gen_{safe}.json"


def _load_cache(model: str) -> dict:
    p = _cache_path(model)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_cache(model: str, cache: dict) -> None:
    _cache_path(model).write_text(json.dumps(cache, indent=2))


def _key(question: str) -> str:
    # Fold the system prompt into the key so a baseline run (no prompt) and a
    # guardrail run (with prompt) produce SEPARATE cache entries instead of one
    # silently returning the other's answers.
    sp = config.SYSTEM_PROMPT or ""
    return hashlib.sha256((sp + "||" + question).encode()).hexdigest()[:16]


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(8))
def chat(model: str, question: str) -> str:
    """The ONLY provider-specific function. Change this to swap providers."""
    kwargs = {
        "model": model,
        "max_tokens": config.GEN_MAX_TOKENS,
        "temperature": config.GEN_TEMPERATURE,
        "messages": [{"role": "user", "content": question}],
    }
    if config.SYSTEM_PROMPT:
        kwargs["system"] = config.SYSTEM_PROMPT   # Claude takes system separately
    resp = _client.messages.create(**kwargs)
    # Claude returns a list of content blocks; join the text ones.
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def stream_chat(model: str, question: str) -> Iterator[str]:
    """Yield text deltas as the model generates, for real-time risk monitoring.

    Provider-specific, like chat(). Not cached: streaming is for the live path.
    """
    kwargs = {
        "model": model,
        "max_tokens": config.GEN_MAX_TOKENS,
        "temperature": config.GEN_TEMPERATURE,
        "messages": [{"role": "user", "content": question}],
    }
    if config.SYSTEM_PROMPT:
        kwargs["system"] = config.SYSTEM_PROMPT
    with _client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            if text:
                yield text


def generate_all(model: str, items: list[dict]) -> dict[str, str]:
    """Return {question: answer} for every item, using/updating the cache."""
    cache = _load_cache(model)
    for i, item in enumerate(items):
        k = _key(item["question"])
        if k not in cache:
            cache[k] = chat(model, item["question"])
            if i % 10 == 0:
                _save_cache(model, cache)
                print(f"  [{model}] generated {i+1}/{len(items)}")
    _save_cache(model, cache)
    return {item["question"]: cache[_key(item["question"])] for item in items}


# --- Sampling for self-consistency checks --------------------------------
# We draw several answers at a non-zero temperature and measure how much they
# agree (see src/evaluators/self_consistency.py). Caching the whole sample set
# keyed by (system prompt, question, n, temperature) keeps re-runs free while
# still varying answers within a single draw.

@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(8))
def _sample_one(model: str, question: str, temperature: float) -> str:
    kwargs = {
        "model": model,
        "max_tokens": config.GEN_MAX_TOKENS,
        "temperature": temperature,
        "messages": [{"role": "user", "content": question}],
    }
    if config.SYSTEM_PROMPT:
        kwargs["system"] = config.SYSTEM_PROMPT
    resp = _client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _samples_cache_path(model: str) -> Path:
    safe = model.replace("/", "_")
    return config.CACHE_DIR / f"samples_{safe}.json"


def sample(model: str, question: str, n: int = 5, temperature: float = 0.7) -> list[str]:
    """Return `n` independently sampled answers for one question (cached)."""
    sp = config.SYSTEM_PROMPT or ""
    key = hashlib.sha256(
        f"{sp}||{question}||n={n}||t={temperature}".encode()
    ).hexdigest()[:16]
    path = _samples_cache_path(model)
    cache = json.loads(path.read_text()) if path.exists() else {}
    if key not in cache:
        cache[key] = [_sample_one(model, question, temperature) for _ in range(n)]
        path.write_text(json.dumps(cache, indent=2))
    return cache[key]
