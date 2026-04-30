from __future__ import annotations

from obliqaxref.curate.judge.run import aggregate_judge_passes, judge_response_from_json
from obliqaxref.curate.judge.schema import QPDecision, QPReasonCode


def _v2_payload(**overrides):
    payload = {
        "judge_schema_version": "v2",
        "passed": True,
        "final_score": 8,
        "subscores": {
            "realism": 8,
            "source_relevance": 9,
            "target_relevance": 9,
            "source_insufficiency": 8,
            "target_necessity": 8,
            "answer_support": 8,
        },
        "binary_checks": {
            "source_relevant": True,
            "target_relevant": True,
            "source_alone_sufficient": False,
            "target_alone_sufficient": False,
            "target_adds_essential_information": True,
            "answer_supported": True,
            "citation_dependent": True,
        },
        "reasons": ["clear citation dependency"],
        "decision_qp": "PASS_QP",
        "reason_code_qp": None,
        "confidence": 0.86,
        "source_alone_insufficient": True,
    }
    payload.update(overrides)
    return payload


def test_v2_judge_payload_passes_when_all_required_checks_hold():
    response = judge_response_from_json("q1", _v2_payload(), pass_threshold=7)

    assert response.decision_qp == QPDecision.PASS_QP
    assert response.passed is True
    assert response.final_score == 8
    assert response.source_alone_sufficient is False
    assert response.target_adds_essential_information is True
    assert response.answer_supported_by_judge is True
    assert response.citation_dependent is True


def test_v2_judge_payload_fails_when_source_alone_is_sufficient():
    payload = _v2_payload(
        passed=True,
        binary_checks={
            "source_relevant": True,
            "target_relevant": True,
            "source_alone_sufficient": True,
            "target_alone_sufficient": False,
            "target_adds_essential_information": True,
            "answer_supported": True,
            "citation_dependent": False,
        },
    )

    response = judge_response_from_json("q1", payload, pass_threshold=7)

    assert response.decision_qp == QPDecision.DROP_QP
    assert response.reason_code_qp == QPReasonCode.QP_NOT_CIT_DEP
    assert response.passed is False
    assert response.source_alone_sufficient is True
    assert response.citation_dependent is False


def test_v2_judge_payload_fails_when_score_below_threshold():
    response = judge_response_from_json(
        "q1",
        _v2_payload(final_score=6, confidence=0.6),
        pass_threshold=7,
    )

    assert response.decision_qp == QPDecision.DROP_QP
    assert response.passed is False


def test_legacy_judge_payload_still_parses():
    response = judge_response_from_json(
        "q1",
        {
            "decision_qp": "PASS_QP",
            "reason_code_qp": None,
            "confidence": 0.9,
            "source_alone_insufficient": True,
        },
    )

    assert response.decision_qp == QPDecision.PASS_QP
    assert response.judge_schema_version == "v1"
    assert response.source_alone_sufficient is False


def test_aggregation_forces_drop_on_v2_failed_answer_support():
    passing = judge_response_from_json("q1", _v2_payload(), pass_threshold=7)
    unsupported = judge_response_from_json(
        "q1",
        _v2_payload(
            binary_checks={
                "source_relevant": True,
                "target_relevant": True,
                "source_alone_sufficient": False,
                "target_alone_sufficient": False,
                "target_adds_essential_information": True,
                "answer_supported": False,
                "citation_dependent": True,
            }
        ),
        pass_threshold=7,
    )

    aggregated = aggregate_judge_passes("q1", [passing, unsupported])

    assert aggregated.decision_qp_final == QPDecision.DROP_QP
    assert aggregated.failure_flags["unsupported_answer"] is True
    assert aggregated.answer_supported_by_judge is False
