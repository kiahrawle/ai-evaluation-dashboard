import src.benchmark_data.cache as cache


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    recs = [{"a": 1}, {"a": 2}]
    cache.save_cached("x", "test", recs)
    assert cache.load_cached("x", "test") == recs
    assert cache.load_cached("missing", "test") is None


def test_cached_or_download_prefers_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.save_cached("x", "test", [{"a": 1}, {"a": 2}, {"a": 3}])
    calls = []

    def downloader():
        calls.append(1)
        return [{"a": 99}]

    out = cache.cached_or_download("x", "test", downloader, limit=2)
    assert out == [{"a": 1}, {"a": 2}]   # sliced to limit
    assert calls == []                    # downloader NOT called


def test_cached_or_download_downloads_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = []

    def downloader():
        calls.append(1)
        return [{"a": 1}, {"a": 2}]

    out = cache.cached_or_download("absent", "test", downloader, limit=1)
    assert calls == [1]
    assert out == [{"a": 1}]
    # The warning points the user at the download script.
    assert "download_benchmarks" in capsys.readouterr().out


def test_split_keys_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.save_cached("d", "train", [{"s": "train"}])
    cache.save_cached("d", "test", [{"s": "test"}])
    assert cache.load_cached("d", "train") == [{"s": "train"}]
    assert cache.load_cached("d", "test") == [{"s": "test"}]
