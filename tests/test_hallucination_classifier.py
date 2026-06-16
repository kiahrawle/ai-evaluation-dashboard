from src.evaluators.hallucination import classify_hallucination


def test_classifier_confidence_and_type():
    item = {
        "question": "Is chocolate poisonous to dogs?",
        "correct_answers": ["Yes, chocolate can be toxic to dogs."],
        "incorrect_answers": ["No, it's fine for dogs."],
    }
    ans = "Yes, chocolate is definitely poisonous to dogs."
    out = classify_hallucination(item, ans)
    assert "severity" in out and 0.0 <= out["severity"] <= 1.0
    assert "confidence" in out and 0.0 <= out["confidence"] <= 1.0
    assert "type" in out

def test_classifier_fake_citation():
    item = {
        "question": "When was the moon formed?",
        "correct_answers": ["About 4.5 billion years ago."],
        "incorrect_answers": ["It formed last week."],
    }
    ans = "According to study [1], it formed 2 years ago."
    out = classify_hallucination(item, ans)
    assert out["type"] == "fake_citation"
