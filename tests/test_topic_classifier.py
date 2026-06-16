from src.utils.topic_classifier import detect_topic, detect_topic_keyword_based


def test_detect_topic_keyword_based():
    medical_text = "What is the treatment for diabetes? I have been experiencing symptoms."
    result = detect_topic_keyword_based(medical_text)
    assert result[0] == "medical"
    assert result[1] > 0.0


def test_detect_topic_finance():
    finance_text = "Should I invest in stocks? What about my portfolio?"
    result = detect_topic_keyword_based(finance_text)
    assert result[0] == "finance"


def test_detect_topic_legal():
    legal_text = "What are my rights? Do I need an attorney for this lawsuit?"
    result = detect_topic_keyword_based(legal_text)
    assert result[0] == "legal"


def test_detect_topic_hybrid():
    text = "What is the biological basis of evolution?"
    result = detect_topic(text, method="hybrid")
    assert "topic" in result
    assert "confidence" in result
    assert result["topic"] in ["medical", "finance", "legal", "science", "politics", "creative", "general"]
