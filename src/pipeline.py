"""Orchestrate generate -> score for a single model."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config
from src import models, scoring
from src.core import EvaluationResult
from src import rag, evaluators, risk, guardrails
from src.utils import scan_for_risk_markers, detect_topic


def evaluate_model(model_name: str, items: list[dict]) -> list[dict]:
    print(f"Generating answers with {model_name} ...")
    answers = models.generate_all(model_name, items)

    print(f"Scoring {model_name} ...")
    results = []
    for i, item in enumerate(items):
        ans = answers[item["question"]]
        # Use multi-judge evaluation; returns per-judge verdicts and agreement metrics
        multi = scoring.evaluate_judges(item, ans, judges=getattr(config, "JUDGE_MODELS", None))
        verdicts = multi.get("verdicts", {})
        agreement = multi.get("agreement_rate", 1.0)
        disagreement = multi.get("disagreement_score", 0.0)
        # Choose majority truthful as the primary 'verdict'
        truths = [bool(v.get("truthful", True)) for v in verdicts.values()] if verdicts else [True]
        primary_truthful = sum(1 for t in truths if t) >= (len(truths) / 2)
        # For informative, take majority as well
        infos = [bool(v.get("informative", True)) for v in verdicts.values()] if verdicts else [True]
        primary_informative = sum(1 for t in infos if t) >= (len(infos) / 2)
        sem = scoring.semantic_truthful(item, ans)
        # Retrieve supporting documents (RAG) for groundedness checks.
        docs = rag.retrieve(item["question"]) or []
        # Compute evaluator scores
        hall_info = evaluators.classify_hallucination(item, ans)
        hall_score = hall_info.get("severity", 0.0)
        confidence_score = hall_info.get("confidence")
        failure_type = hall_info.get("type")
        # Claim-level decomposition: label each atomic claim against the
        # reference sets so we can see WHERE an answer goes wrong.
        claim_info = evaluators.score_claims(item, ans)
        # Optional reference-free signal: do independent samples agree?
        self_consistency = None
        if getattr(config, "SELF_CONSISTENCY_ENABLED", False):
            self_consistency = evaluators.self_consistency_score(
                model_name, item["question"],
                n=getattr(config, "SELF_CONSISTENCY_N", 5),
            )
        ground_score = evaluators.score_groundedness(docs, ans)
        # Verify citation support (links each claim to its best supporting doc).
        citation_info = evaluators.verify_citation_support(ans, docs)
        # Numeric contradiction: catch "docs say 12M, answer says 50M" that pure
        # embedding similarity would wave through as supported.
        contradiction_info = evaluators.detect_numeric_contradictions(ans, docs)
        if contradiction_info.get("contradiction_detected"):
            # A factual contradiction means the answer is NOT grounded, no matter
            # how topically similar it looks.
            citation_info["citation_supported"] = False
        unsupported_ratio = 1.0 - (citation_info.get("support_score") or 0.5)
        # Dangerous-hallucination signal: severity weighted by confidence
        # (reuse hall_info rather than re-classifying).
        dangerous_score = float(hall_info.get("dangerous_score") or 0.0)
        dangerous_level = ("HIGH" if dangerous_score >= 0.5
                           else "MEDIUM" if dangerous_score >= 0.25 else "LOW")
        
        # Scan for risk markers: uncertainty, risky phrases, contradictions, citation gaps
        risk_markers = scan_for_risk_markers(ans)
        contradiction_score = risk_markers.get("contradictions", {}).get("contradiction_score", 0.0)
        
        # Detect topic for adaptive guardrails
        topic_detection = detect_topic(item["question"], method="hybrid")
        detected_topic = topic_detection.get("topic", "general")
        topic_confidence = topic_detection.get("confidence", 0.0)
        
        # Get adaptive guardrail settings for this topic
        adaptive_settings = guardrails.adaptive_guardrails_summary(detected_topic)
        
        # Score refusal quality if this is a refusal
        refusal_quality = evaluators.score_refusal_quality(ans)
        # Compute comprehensive risk score with weighted formula
        risk_details = risk.score_risk_with_details(
            hallucination_score=hall_score,
            grounded_score=ground_score,
            confidence_score=confidence_score,
            unsupported_claims_ratio=unsupported_ratio,
            contradiction_score=contradiction_score,
        )
        risk_score = risk_details["risk_score"]
        risk_level = risk_details["risk_level"]
        
        # Determine interventions based on risk
        interventions = guardrails.recommend_interventions(risk_details)
        recommended_actions = interventions.get("recommended_interventions", [])
        intervention_rationale = interventions.get("rationale", {})
        # Build an EvaluationResult instance and export as dict for
        # backward-compatible downstream processing.
        er = EvaluationResult(
            prompt=item["question"],
            response=ans,
            retrieved_docs=docs,
            truthful=bool(primary_truthful),
            informative=bool(primary_informative),
            grounded=bool(sem),
            hallucination_score=hall_score,
            risk_score=risk_score,
            confidence_score=confidence_score,
            guardrail_triggered=bool(config.SYSTEM_PROMPT),
            failure_type=failure_type or "",
            model=model_name,
            metadata={
                "category": item.get("category"),
                "judge_votes": verdicts,
                "judge_agreement": agreement,
                "judge_disagreement": disagreement,
                "citation_supported": citation_info.get("citation_supported"),
                "supported_claims": citation_info.get("supported_claims"),
                "total_claims": citation_info.get("total_claims"),
                "support_score": citation_info.get("support_score"),
                "unsupported_claims": citation_info.get("unsupported_claims", []),
                "risk_level": risk_level,
                "risk_markers": {
                    "uncertainty": risk_markers.get("uncertainty", {}),
                    "risky_phrases": risk_markers.get("risky_phrases", {}),
                    "contradictions": risk_markers.get("contradictions", {}),
                    "citation_gaps": risk_markers.get("citation_gaps", {}),
                },
                "recommended_interventions": recommended_actions,
                "intervention_rationale": intervention_rationale,
                "detected_topic": detected_topic,
                "topic_confidence": topic_confidence,
                "adaptive_guardrail_level": adaptive_settings.get("enforcement_level"),
                "require_citations": adaptive_settings.get("require_citations"),
                "refusal_is_refusal": refusal_quality.get("is_refusal"),
                "refusal_helpfulness": refusal_quality.get("helpfulness"),
                "refusal_explanation_quality": refusal_quality.get("explanation_quality"),
                "refusal_educational_value": refusal_quality.get("educational_value"),
                "refusal_overall_quality": refusal_quality.get("overall_quality"),
                "is_refusal_classifier": hall_info.get("is_refusal"),
                "claim_total": claim_info.get("total_claims"),
                "claim_supported": claim_info.get("supported"),
                "claim_contradicted": claim_info.get("contradicted"),
                "claim_unsupported": claim_info.get("unsupported"),
                "claim_hallucination_rate": claim_info.get("claim_hallucination_rate"),
                "claims": claim_info.get("claims", []),
                "self_consistency": (
                    {k: v for k, v in self_consistency.items() if k != "samples"}
                    if self_consistency else None
                ),
                "dangerous_score": dangerous_score,
                "dangerous_level": dangerous_level,
                "citation_evidence": citation_info.get("evidence", []),
                "numeric_contradiction": contradiction_info.get("contradiction_detected"),
                "numeric_contradictions": contradiction_info.get("contradictions", []),
            },
        )
        # Mark guardrail triggers based on computed scores (post-hoc).
        if guardrails.should_trigger(er.to_dict()):
            er.guardrail_triggered = True
        results.append(er.to_dict())
        if i % 25 == 0:
            print(f"  scored {i+1}/{len(items)}")
    return results
