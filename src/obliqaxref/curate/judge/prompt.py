# src/obliqaxref/curate/judge/prompt.py
"""
Judge prompts: QP-only (answer-agnostic) judge prompt with strict JSON output contract.

Defines the system and user prompts for the judge LLM to assess question-passage
alignment and citation dependence WITHOUT validating the gold answer.

Design:
- Answer-agnostic: Does NOT reference or validate gold_answer
- Focuses on: citation dependence, target necessity, question quality
- Conservative: Prefer DROP_QP with reason code when uncertain
"""

from __future__ import annotations

from typing import Any

QP_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for STRICTLY CITATION-DEPENDENT regulatory QA validation.

You will be given:
- SOURCE passage
- TARGET passage
- QUESTION
- GOLD ANSWER

This is a POST-IR quality check for a STRICTLY CITATION-DEPENDENT benchmark.

STRICT REQUIREMENT: the item must genuinely require following the source-to-target cross-reference.
Be conservative. Borderline cases fail unless citation dependency is clear.

PASS only if ALL are true:
1) SOURCE is relevant to the question.
2) TARGET is relevant to the question.
3) SOURCE alone is insufficient for a complete and correct answer.
4) TARGET adds essential missing information.
5) The item is citation-dependent: it requires using SOURCE context plus TARGET detail.
6) The GOLD ANSWER is supported by SOURCE and TARGET, with no material hallucination.
7) The question is realistic and well-formed for regulatory/compliance use.

TARGET-alone check:
- If TARGET alone answers the question without needing SOURCE-side regulatory context, set target_alone_sufficient=true.
- target_alone_sufficient=true does not automatically fail, but it is evidence against citation dependency unless SOURCE still provides essential scope, actor, condition, or cross-reference context.

Legacy compatibility:
- Also set decision_qp = PASS_QP when passed=true, otherwise DROP_QP.
- If source_alone_sufficient=true, decision_qp MUST be DROP_QP with reason_code_qp="QP_NOT_CIT_DEP".
- If target_adds_essential_information=false or target is irrelevant, use reason_code_qp="QP_WRONG_TARGET".
- If answer_supported=false, use reason_code_qp="QP_ILL_FORMED" unless another reason is clearly better.

Example PASS_QP:
- SOURCE: "Firms must maintain capital adequacy..." (mentions requirement but NO specifics)
- TARGET: "...minimum tier-1 capital ratio of 8.5%..." (provides the REQUIRED missing detail)
- QUESTION: "What is the minimum tier-1 capital ratio?"
- Result: PASS_QP (source insufficient, target provides required detail)

DROP_QP only for clear failures, with ONE reason code:
- QP_NOT_CIT_DEP: SOURCE alone FULLY answers the question (not citation-dependent; STRICT FILTER)
- QP_WRONG_TARGET: TARGET is not actually relevant or provides no usable supporting detail.
- QP_SCOPE_MISMATCH: The QUESTION's actor/regime/condition clearly conflicts with the passages.
- QP_UNDER_SPEC: The QUESTION is too ambiguous to judge with these passages.
- QP_TOO_BROAD: The QUESTION is multi-part or too general for these two passages.
- QP_ILL_FORMED: QUESTION is unclear or not evaluable.

Example DROP_QP with QP_NOT_CIT_DEP:
- SOURCE: "All directors must be disclosed with their experience in the annual report."
- TARGET: "John Smith has 20 years experience..." (adds specifics but source already answers)
- QUESTION: "What board information must be disclosed?"
- Result: DROP_QP with QP_NOT_CIT_DEP (source alone sufficient, not citation-dependent)

Output MUST be a single JSON object and nothing else (no Markdown, no code fences).
Schema:
{
  "judge_schema_version": "v2",
  "passed": true or false,
  "final_score": <integer 0-10>,
  "subscores": {
    "realism": <integer 0-10>,
    "source_relevance": <integer 0-10>,
    "target_relevance": <integer 0-10>,
    "source_insufficiency": <integer 0-10>,
    "target_necessity": <integer 0-10>,
    "answer_support": <integer 0-10>
  },
  "binary_checks": {
    "source_relevant": <bool>,
    "target_relevant": <bool>,
    "source_alone_sufficient": <bool>,
    "target_alone_sufficient": <bool>,
    "target_adds_essential_information": <bool>,
    "answer_supported": <bool>,
    "citation_dependent": <bool>
  },
  "reasons": [<short strings explaining failures or important support>],
  "decision_qp": "PASS_QP" or "DROP_QP",
  "reason_code_qp": <required if DROP_QP; null if PASS_QP>,
  "confidence": <float 0.0–1.0>,
  "source_alone_insufficient": <bool; true if source alone cannot answer>,
  "target_contains_missing_detail": <optional bool>,
  "question_well_formed": <optional bool>,
  "key_missing_detail": <optional; what TARGET contributes>,
  "notes": <optional brief explanation>
}

CRITICAL: If source_alone_insufficient=false, you MUST set decision_qp="DROP_QP" with reason_code_qp="QP_NOT_CIT_DEP".
Only truly citation-dependent items should pass.
"""


def build_qp_judge_prompt(
    question: str,
    source_text: str,
    target_text: str,
    gold_answer: str | None = None,
    source_passage_id: str | None = None,
    target_passage_id: str | None = None,
) -> str:
    """
    Build the QP-only user prompt for judge LLM evaluation (answer-agnostic).

    Args:
        question: The question to evaluate
        source_text: The source passage text
        target_text: The target passage text
        source_passage_id: Optional source passage identifier (for traceability)
        target_passage_id: Optional target passage identifier (for traceability)

    Returns:
        Formatted prompt string (no answer reference)
    """
    source_label = f" (id={source_passage_id})" if source_passage_id else ""
    target_label = f" (id={target_passage_id})" if target_passage_id else ""

    prompt = f"""SOURCE PASSAGE{source_label}:
{source_text}

TARGET PASSAGE{target_label}:
{target_text}

QUESTION:
{question}

GOLD ANSWER:
{gold_answer or ""}

Task: Apply the strict v2 rubric. The item passes only when SOURCE and TARGET are both relevant, SOURCE alone is insufficient, TARGET adds essential missing information, the answer is supported, and the item genuinely requires following the source-to-target cross-reference.
Return ONLY a single JSON object."""

    return prompt


def get_qp_json_schema() -> dict[str, Any]:
    """
    Return the JSON schema for QP-only structured output.

    This schema enforces the judge LLM to respond in the exact format expected,
    including required reason codes for DROP_QP decisions.
    """
    return {
        "type": "object",
        "properties": {
            "decision_qp": {
                "type": "string",
                "enum": ["PASS_QP", "DROP_QP"],
                "description": "QP validation decision",
            },
            "reason_code_qp": {
                "type": ["string", "null"],
                "enum": [
                    "QP_NOT_CIT_DEP",
                    "QP_WRONG_TARGET",
                    "QP_UNDER_SPEC",
                    "QP_SCOPE_MISMATCH",
                    "QP_TOO_BROAD",
                    "QP_ILL_FORMED",
                    None,
                ],
                "description": "Required if DROP_QP, identifies the failure reason",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score from 0.0 to 1.0",
            },
            "answerable_from_source_only": {
                "type": ["boolean", "null"],
                "description": "Optional: Is the question answerable from source passage alone?",
            },
            "target_contains_missing_detail": {
                "type": ["boolean", "null"],
                "description": "Optional: Does the target passage contain the missing detail?",
            },
            "question_well_formed": {
                "type": ["boolean", "null"],
                "description": "Optional: Is the question well-formed and in-scope?",
            },
            "key_missing_detail": {
                "type": ["string", "null"],
                "description": "Optional: What detail the target passage provides",
            },
            "support_snippets": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional: List of short spans prefixed with SOURCE: or TARGET:",
            },
            "notes": {
                "type": ["string", "null"],
                "description": "Optional: Brief explanation of the decision",
            },
            "judge_schema_version": {"type": ["string", "null"]},
            "passed": {"type": ["boolean", "null"]},
            "final_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 10},
            "subscores": {"type": ["object", "null"]},
            "binary_checks": {"type": ["object", "null"]},
            "reasons": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "required": ["decision_qp", "confidence"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"decision_qp": {"const": "DROP_QP"}}},
                "then": {
                    "required": ["reason_code_qp"],
                    "properties": {"reason_code_qp": {"type": "string"}},
                },
            },
            {
                "if": {"properties": {"decision_qp": {"const": "PASS_QP"}}},
                "then": {
                    "properties": {"reason_code_qp": {"type": "null"}},
                },
            },
        ],
    }
