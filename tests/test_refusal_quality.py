from src.evaluators.refusal_quality import (
    detect_is_refusal,
    score_refusal_quality,
    score_refusal_helpfulness,
)


def test_detect_is_refusal():
    assert detect_is_refusal("I cannot answer that question.") is True
    assert detect_is_refusal("I'm not able to provide that information.") is True
    assert detect_is_refusal("The answer is 42.") is False


def test_score_refusal_helpfulness():
    # High quality refusal with redirection
    good_refusal = (
        "I cannot provide medical advice. However, here's what I can help with: "
        "I can explain general health concepts. I'd recommend consulting a doctor for diagnosis."
    )
    score = score_refusal_helpfulness(good_refusal)
    assert score > 0.5
    
    # Low quality bare refusal
    bad_refusal = "I can't do that."
    score = score_refusal_helpfulness(bad_refusal)
    assert score < 0.5


def test_score_refusal_quality_comprehensive():
    # Helpful, educational refusal
    refusal = (
        "I cannot provide investment advice. Here's why: as an AI, I lack personal knowledge of your "
        "financial situation. Instead, I can explain basic investment concepts. You should consult a "
        "qualified financial advisor. I recommend checking resources like the SEC website."
    )
    result = score_refusal_quality(refusal)
    assert result["is_refusal"] is True
    assert result["helpfulness"] > 0.4
    assert result["explanation_quality"] > 0.3
    assert result["educational_value"] > 0.3
    assert result["overall_quality"] > 0.3


def test_score_refusal_quality_non_refusal():
    # Non-refusal response
    response = "The capital of France is Paris."
    result = score_refusal_quality(response)
    assert result["is_refusal"] is False
    assert result["helpfulness"] is None
