"""Offline tests for the benchmark loaders: exercise the pure `_normalize`
functions against representative HF rows (no network)."""
from src.datasets import mmlu, gsm8k, bbq, halueval, toxicity, registry


def test_mmlu_normalize():
    row = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1, "subject": "math"}
    rec = mmlu._normalize(row)
    assert rec["task"] == "multiple_choice"
    assert rec["choices"][1] == "4" and rec["answer"] == 1
    assert rec["category"] == "math"


def test_gsm8k_normalize_parses_gold():
    row = {"question": "How many?", "answer": "work...\n#### 1,234"}
    rec = gsm8k._normalize(row)
    assert rec["task"] == "numeric"
    assert rec["answer"] == "1234"


def test_bbq_normalize_builds_mc():
    row = {"context": "At the bank.", "question": "Who was nervous?",
           "ans0": "the old man", "ans1": "the young man", "ans2": "unknown",
           "label": 2, "category": "Age"}
    rec = bbq._normalize(row)
    assert rec["task"] == "multiple_choice"
    assert rec["choices"] == ["the old man", "the young man", "unknown"]
    assert rec["answer"] == 2
    assert "At the bank." in rec["question"]


def test_halueval_normalize_emits_two_balanced_records():
    row = {"knowledge": "Paris is the capital of France.",
           "question": "Capital of France?",
           "right_answer": "Paris", "hallucinated_answer": "Lyon"}
    recs = halueval._normalize(row)
    assert len(recs) == 2
    labels = sorted(r["answer"] for r in recs)
    assert labels == [False, True]
    assert all(r["task"] == "hallucination_detection" for r in recs)


def test_toxicity_normalize():
    row = {"prompt": {"text": "Some prompt", "toxicity": 0.1}, "challenging": True}
    rec = toxicity._normalize(row)
    assert rec["task"] == "toxicity_generation"
    assert rec["question"] == "Some prompt"
    assert rec["meta"]["prompt_toxicity"] == 0.1


def test_registry_available_and_tasks():
    avail = registry.available()
    for name in ["truthfulqa", "halueval", "mmlu", "gsm8k", "bbq", "toxicity"]:
        assert name in avail
    assert registry.TASKS["mmlu"] == "multiple_choice"
    assert registry.TASKS["gsm8k"] == "numeric"


def test_load_benchmark_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        registry.load_benchmark("not_a_dataset")
