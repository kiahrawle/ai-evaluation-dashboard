"""Load TruthfulQA into a simple list of dicts.

Schema per item:
    question          str
    category          str   (one of 38 topics: Health, Law, Finance, ...)
    best_answer       str
    correct_answers   list[str]
    incorrect_answers list[str]
"""
from datasets import load_dataset


def load_truthfulqa(limit: int | None = None) -> list[dict]:
    # The "generation" config is the free-text task. Split is "validation".
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    rows = []
    for r in ds:
        rows.append(
            {
                "question": r["question"],
                "category": r["category"],
                "best_answer": r["best_answer"],
                "correct_answers": r["correct_answers"],
                "incorrect_answers": r["incorrect_answers"],
            }
        )
    if limit:
        rows = rows[:limit]
    return rows


def coarse_category(fine: str) -> str:
    """Optional: map the 38 fine categories into broader buckets so you can
    report 'health/legal vs trivia' style cuts. Tune this mapping to taste."""
    high_stakes = {"Health", "Law", "Finance", "Nutrition", "Psychology"}
    trivia = {"Misconceptions", "Sociology", "History", "Language",
              "Confusion: People", "Confusion: Places", "Statistics"}
    if fine in high_stakes:
        return "high_stakes"
    if fine in trivia:
        return "trivia"
    return "other"
