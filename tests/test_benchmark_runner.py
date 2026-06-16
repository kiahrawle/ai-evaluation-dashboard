"""Offline test of the generic benchmark runner with a stubbed generate_fn."""
from src.benchmark_data.runner import run_records


def test_run_records_multiple_choice_accuracy():
    records = [
        {"dataset": "mmlu", "task": "multiple_choice", "question": "q1",
         "choices": ["a", "b"], "answer": 0, "category": "x"},
        {"dataset": "mmlu", "task": "multiple_choice", "question": "q2",
         "choices": ["a", "b"], "answer": 1, "category": "x"},
    ]
    # Always answers "A" -> correct on the first, wrong on the second.
    out = run_records(records, generate_fn=lambda prompt: "A")
    assert out["n"] == 2
    assert out["accuracy"] == 0.5
    assert out["task"] == "multiple_choice"


def test_run_records_numeric():
    records = [{"dataset": "gsm8k", "task": "numeric", "question": "q",
                "choices": None, "answer": "42", "category": "math"}]
    out = run_records(records, generate_fn=lambda prompt: "the answer is 42")
    assert out["accuracy"] == 1.0


def test_run_records_toxicity_aggregates():
    records = [
        {"dataset": "toxicity", "task": "toxicity_generation", "question": "p1", "category": "t"},
        {"dataset": "toxicity", "task": "toxicity_generation", "question": "p2", "category": "t"},
    ]
    # Inject a deterministic backend so the metric is meaningful in the test.
    backend = lambda text: 0.9 if "bad" in text else 0.0
    out = run_records(records, generate_fn=lambda p: "bad" if p == "p1" else "fine",
                      toxicity_backend=backend)
    assert out["n"] == 2
    assert out["mean_toxicity"] == 0.45
    assert out["toxic_rate"] == 0.5
