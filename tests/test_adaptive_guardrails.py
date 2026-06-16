from src.guardrails.adaptive import (
    get_guardrail_for_topic,
    get_enforcement_level,
    should_apply_strict_checking,
    should_require_citations,
    create_adaptive_system_prompt,
    adaptive_guardrails_summary,
)


def test_get_guardrail_for_topic():
    medical_guardrail = get_guardrail_for_topic("medical")
    assert "doctor" in medical_guardrail.lower()
    assert "healthcare" in medical_guardrail.lower()
    
    finance_guardrail = get_guardrail_for_topic("finance")
    assert "advisor" in finance_guardrail.lower() or "financial" in finance_guardrail.lower()
    
    creative_guardrail = get_guardrail_for_topic("creative")
    assert "creative" in creative_guardrail.lower() or "imaginative" in creative_guardrail.lower()


def test_enforcement_levels():
    assert get_enforcement_level("medical") == "strict"
    assert get_enforcement_level("finance") == "strict"
    assert get_enforcement_level("legal") == "strict"
    assert get_enforcement_level("science") == "moderate"
    assert get_enforcement_level("creative") == "lenient"


def test_should_apply_strict_checking():
    assert should_apply_strict_checking("medical") is True
    assert should_apply_strict_checking("creative") is False
    assert should_apply_strict_checking("general") is False


def test_should_require_citations():
    assert should_require_citations("medical") is True
    assert should_require_citations("science") is True
    assert should_require_citations("creative") is False


def test_adaptive_guardrails_summary():
    summary = adaptive_guardrails_summary("medical")
    assert summary["topic"] == "medical"
    assert summary["enforcement_level"] == "strict"
    assert summary["strict_checking"] is True
    assert summary["require_citations"] is True
    assert "guardrail_prompt" in summary
