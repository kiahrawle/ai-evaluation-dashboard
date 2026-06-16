from src.evaluators.tasks import (
    extract_choice, score_multiple_choice,
    extract_final_number, score_numeric,
    score_toxicity, score_record, format_prompt,
)


# --- multiple choice ---------------------------------------------------------
def test_extract_choice_explicit_letter():
    assert extract_choice("The answer is (C).", ["a", "b", "c", "d"]) == 2
    assert extract_choice("answer: B", ["a", "b", "c"]) == 1
    assert extract_choice("B", ["a", "b", "c"]) == 1


def test_extract_choice_by_text():
    assert extract_choice("I'd say Paris.", ["London", "Paris", "Rome"]) == 1


def test_extract_choice_none_when_ambiguous():
    assert extract_choice("not sure", ["x", "y"]) is None


def test_score_multiple_choice():
    rec = {"task": "multiple_choice", "choices": ["a", "b", "c", "d"], "answer": 2}
    assert score_multiple_choice("The answer is C", rec) is True
    assert score_multiple_choice("A", rec) is False


# --- numeric -----------------------------------------------------------------
def test_extract_final_number():
    assert extract_final_number("So the total is 42.") == 42.0
    assert extract_final_number("steps... #### 1,234") == 1234.0
    assert extract_final_number("it costs $3.50 total") == 3.5
    assert extract_final_number("no digits here") is None


def test_score_numeric():
    assert score_numeric("the answer is 18", 18) is True
    assert score_numeric("answer: 18", "#### 18") is True
    assert score_numeric("answer: 19", 18) is False


# --- toxicity ----------------------------------------------------------------
def test_score_toxicity_lexical_and_backend():
    assert score_toxicity("have a lovely day") == 0.0
    assert score_toxicity("you are an idiot and i hate you") > 0.0
    # Backend overrides the lexical heuristic.
    assert score_toxicity("anything", backend=lambda t: 0.9) == 0.9


# --- format + dispatch -------------------------------------------------------
def test_format_prompt_multiple_choice_includes_options():
    rec = {"task": "multiple_choice", "question": "Capital of France?",
           "choices": ["London", "Paris"]}
    prompt = format_prompt(rec)
    assert "A. London" in prompt and "B. Paris" in prompt


def test_score_record_dispatch():
    mc = {"task": "multiple_choice", "choices": ["a", "b"], "answer": 1}
    assert score_record(mc, "B")["correct"] is True

    num = {"task": "numeric", "answer": "7"}
    assert score_record(num, "the answer is 7")["correct"] is True

    halluc = {"task": "hallucination_detection", "answer": True}
    assert score_record(halluc, "This is hallucinated.")["correct"] is True
    assert score_record(halluc, "faithful")["correct"] is False

    tox = {"task": "toxicity_generation"}
    out = score_record(tox, "i hate you, idiot")
    assert out["toxicity"] > 0.0 and "toxic" in out
