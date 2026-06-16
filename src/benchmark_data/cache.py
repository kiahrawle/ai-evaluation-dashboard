"""Local-first cache for benchmark datasets.

Downloading from HuggingFace on every run is slow and breaks offline / CI use.
Loaders call `cached_or_download`: if a normalized cache file exists under
`data/cached_benchmarks/` it is parsed locally; otherwise the dataset is fetched
(with a clear warning) and the caller is told to run the download script to
cache it.

Cache files are keyed by (dataset, split) and hold the normalized records, so
they are small, human-readable, and identical to what the loader would return.
"""
import json
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cached_benchmarks"


def cache_file(name: str, split: str) -> Path:
    return CACHE_DIR / f"{name}__{split}.json"


def load_cached(name: str, split: str) -> Optional[List[Dict[str, Any]]]:
    p = cache_file(name, split)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def save_cached(name: str, split: str, records: List[Dict[str, Any]]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_file(name, split)
    p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def cached_or_download(
    name: str,
    split: str,
    downloader: Callable[[], List[Dict[str, Any]]],
    limit: Optional[int] = None,
    warn: bool = True,
) -> List[Dict[str, Any]]:
    """Return cached records if present, else download (with a warning).

    `downloader` returns the normalized records (it may already apply `limit`);
    cached records are sliced to `limit` here.
    """
    records = load_cached(name, split)
    if records is None:
        if warn:
            print(
                f"[datasets] No local cache for '{name}' (split '{split}') at "
                f"{cache_file(name, split)}.\n"
                f"           Downloading from HuggingFace — run "
                f"`python scripts/download_benchmarks.py` once to cache it for "
                f"offline / CI use."
            )
        records = downloader()
    return records[:limit] if limit else records


__all__ = ["CACHE_DIR", "cache_file", "load_cached", "save_cached", "cached_or_download"]
