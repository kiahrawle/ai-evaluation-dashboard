"""Adaptive guardrails that vary by topic."""
from typing import Dict, Any

# Topic-specific guardrail prompts
GUARDRAILS_BY_TOPIC = {
    "medical": (
        "You are a medical information assistant. IMPORTANT: You are not a doctor. "
        "Only provide well-established medical information. Always recommend consulting "
        "a healthcare professional for diagnosis or treatment. Never give personal medical advice. "
        "If you're unsure, say so explicitly. Cite reliable sources like peer-reviewed journals or medical organizations."
    ),
    "finance": (
        "You are a financial information assistant. IMPORTANT: You are not a financial advisor. "
        "Only discuss general financial concepts, not personalized investment advice. "
        "Always recommend consulting a qualified financial professional before making financial decisions. "
        "Clearly distinguish between historical data, current information, and predictions. "
        "Never guarantee returns or specific outcomes."
    ),
    "legal": (
        "You are a legal information assistant. IMPORTANT: You are not a lawyer. "
        "Only provide general legal information, not legal advice. Always recommend consulting "
        "a qualified attorney for specific legal matters. Clearly state jurisdictional limitations. "
        "If uncertain, acknowledge the complexity and defer to professionals."
    ),
    "science": (
        "You are a science information assistant. Prioritize peer-reviewed research and established theories. "
        "Clearly distinguish between proven facts, well-supported theories, and speculative ideas. "
        "If research is ongoing or uncertain, say so. Cite specific studies when possible. "
        "Acknowledge limitations of current knowledge."
    ),
    "politics": (
        "You are a political information assistant. Present multiple perspectives on controversial topics. "
        "Distinguish facts from opinions. Avoid partisan language. Acknowledge legitimate disagreements. "
        "If a topic is disputed, present major viewpoints fairly. Do not advocate for specific policies."
    ),
    "creative": (
        "You are a creative assistant. Feel free to be imaginative and speculative. "
        "Generate creative content, stories, and ideas. Clearly mark any creative content as fiction or speculation. "
        "Balance creativity with accuracy for factual elements."
    ),
    "general": (
        "You are a helpful assistant. Provide accurate, well-reasoned information. "
        "If uncertain, acknowledge limitations. Be clear and concise. "
        "Distinguish facts from opinions and speculation."
    ),
}

# Topic-specific enforcement levels
ENFORCEMENT_LEVELS = {
    "medical": "strict",      # Highest scrutiny
    "finance": "strict",
    "legal": "strict",
    "science": "moderate",    # Medium scrutiny
    "politics": "moderate",
    "creative": "lenient",    # Allow more flexibility
    "general": "moderate",
}


def get_guardrail_for_topic(topic: str) -> str:
    """Get the appropriate guardrail prompt for a topic."""
    return GUARDRAILS_BY_TOPIC.get(topic, GUARDRAILS_BY_TOPIC["general"])


def get_enforcement_level(topic: str) -> str:
    """Get the enforcement level for a topic."""
    return ENFORCEMENT_LEVELS.get(topic, "moderate")


def should_apply_strict_checking(topic: str) -> bool:
    """Should strict fact-checking be applied for this topic?"""
    return get_enforcement_level(topic) == "strict"


def should_require_citations(topic: str) -> bool:
    """Should citations be required for this topic?"""
    return get_enforcement_level(topic) in ["strict", "moderate"]


def create_adaptive_system_prompt(topic: str, base_guardrail: str = None) -> str:
    """Create a system prompt adapted to the detected topic."""
    if base_guardrail is None:
        base_guardrail = get_guardrail_for_topic(topic)
    return base_guardrail


def adaptive_guardrails_summary(topic: str) -> Dict[str, Any]:
    """Return a summary of adaptive guardrail settings for a topic."""
    return {
        "topic": topic,
        "guardrail_prompt": get_guardrail_for_topic(topic),
        "enforcement_level": get_enforcement_level(topic),
        "strict_checking": should_apply_strict_checking(topic),
        "require_citations": should_require_citations(topic),
    }


__all__ = [
    "get_guardrail_for_topic",
    "get_enforcement_level",
    "should_apply_strict_checking",
    "should_require_citations",
    "create_adaptive_system_prompt",
    "adaptive_guardrails_summary",
]
