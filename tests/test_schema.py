from src.core import EvaluationResult


def test_evaluation_result_roundtrip():
    er = EvaluationResult(
        prompt="What is two plus two?",
        response="4",
        retrieved_docs=[{"text": "2+2=4"}],
        truthful=True,
        informative=True,
        grounded=True,
        hallucination_score=0.0,
        risk_score=0.0,
        confidence_score=0.99,
        guardrail_triggered=False,
        failure_type=None,
        model="test-model",
        metadata={"category": "math"},
    )
    d = er.to_dict()
    er2 = EvaluationResult.from_dict(d)
    assert er2.prompt == er.prompt
    assert er2.response == er.response
    assert er2.retrieved_docs == er.retrieved_docs
    assert er2.model == "test-model"
