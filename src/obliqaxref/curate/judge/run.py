"""
Judge orchestration (Azure-only): Select JUDGE_IR items → build queue → multi-pass LLM → aggregate → write outputs.

Design:
- Strict citation-dependency rubric with answer-support auditing
- Azure-only: uses Azure OpenAI deployments via AzureOpenAI client
- Conservative: prefer DROP_QP on errors/uncertainty
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Optional .env support: explicitly load .env from project root, override existing env vars
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(".env"), override=True)
except Exception:
    pass

from openai import AzureOpenAI

from obliqaxref.config import RunConfig
from obliqaxref.utils.io import ensure_dir, write_json

from .prompt import QP_JUDGE_SYSTEM_PROMPT, build_qp_judge_prompt
from .schema import (
    AggregatedJudgeResponse,
    JUDGE_SCHEMA_VERSION,
    JudgeQueueItem,
    JudgeResponse,
    QPDecision,
    QPReasonCode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers: endpoint normalization and lenient JSON parsing
# ---------------------------------------------------------------------
def _mask(s: str) -> str:
    """Safely mask sensitive strings for logging (show first 4 and last 4 chars)."""
    s = s or ""
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}{'*' * (len(s) - 8)}{s[-4:]}"


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def normalize_azure_endpoint(endpoint: str) -> str:
    """
    Azure endpoint must be: https://<resource>.openai.azure.com
    (no trailing slash, no /openai)
    """
    e = (endpoint or "").strip().rstrip("/")
    if e.endswith("/openai"):
        e = e[:-7]
    return e


def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_first_json_object(s: str) -> str | None:
    if not s:
        return None
    s2 = strip_code_fences(s)
    m = _JSON_BLOCK_RE.search(s2)
    return m.group(0).strip() if m else None


def parse_json_lenient(s: str) -> dict[str, Any] | None:
    if not s:
        return None
    s2 = strip_code_fences(s)
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    obj_txt = extract_first_json_object(s2)
    if not obj_txt:
        return None
    try:
        obj = json.loads(obj_txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def load_judge_items(curated_items_file: Path) -> list[dict[str, Any]]:
    """
    Load items with JUDGE or JUDGE_IR decision flag from curated output.
    Prefers `decision_ir` but falls back to `decision`.
    """
    items: list[dict[str, Any]] = []
    with open(curated_items_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            decision_ir = item.get("decision_ir")
            decision = decision_ir if decision_ir is not None else item.get("decision")
            if decision in {"JUDGE_IR", "JUDGE"}:
                items.append(item)

    logger.info("Loaded %d items for judge evaluation", len(items))
    return items


def build_judge_queue(
    items: list[dict[str, Any]], passages: dict[str, dict[str, Any]]
) -> list[JudgeQueueItem]:
    queue: list[JudgeQueueItem] = []

    for item in items:
        item_id = item.get("item_id")
        question = item.get("question")
        source_pid = item.get("source_passage_id")
        target_pid = item.get("target_passage_id")

        # Guardrails: skip invalid entries conservatively
        if not item_id or not question or not source_pid or not target_pid:
            logger.warning(
                "Skipping item missing required fields: item_id=%s question_present=%s source_pid=%s target_pid=%s",
                item_id,
                bool(question),
                source_pid,
                target_pid,
            )
            continue

        source_text = item.get("source_text") or passages.get(source_pid, {}).get("text", "") or ""
        target_text = item.get("target_text") or passages.get(target_pid, {}).get("text", "") or ""

        queue.append(
            JudgeQueueItem(
                item_id=item_id,
                question=question,
                source_passage_id=source_pid,
                source_text=source_text,
                target_passage_id=target_pid,
                target_text=target_text,
                gold_answer=item.get("gold_answer") or item.get("expected_answer"),
                ir_votes=item.get("ir_votes"),
                metadata=item.get("metadata"),
            )
        )

    logger.info("Built QP judge queue with %d items", len(queue))
    return queue


# ---------------------------------------------------------------------
# Azure LLM call
# ---------------------------------------------------------------------
def build_azure_client(
    *,
    azure_endpoint: str,
    api_key: str,
    api_version: str,
    timeout_s: float = 60.0,
    max_retries: int = 0,
) -> AzureOpenAI:
    # Note: AzureOpenAI internally builds /openai/deployments/... routes
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=timeout_s,
        max_retries=max_retries,
    )


def call_judge_llm_once(
    client: AzureOpenAI,
    *,
    deployment: str,
    queue_item: JudgeQueueItem,
    temperature: float,
    max_completion_tokens: int,
    pass_threshold: int = 7,
) -> JudgeResponse:
    """
    Single-pass QP-only judgement call (answer-agnostic).
    """
    system_prompt = QP_JUDGE_SYSTEM_PROMPT
    user_prompt = build_qp_judge_prompt(
        question=queue_item.question,
        source_text=queue_item.source_text,
        target_text=queue_item.target_text,
        gold_answer=queue_item.gold_answer,
        source_passage_id=queue_item.source_passage_id,
        target_passage_id=queue_item.target_passage_id,
    )

    resp = client.chat.completions.create(
        model=deployment,  # Azure deployment name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,  # IMPORTANT for GPT-5.2 on Azure
    )

    text = (resp.choices[0].message.content or "").strip()
    obj = parse_json_lenient(text)

    if not isinstance(obj, dict):
        raise ValueError(f"Judge response not valid JSON schema. Raw text: {text[:200]}")

    return judge_response_from_json(queue_item.item_id, obj, pass_threshold=pass_threshold)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return None


def _as_score(value: Any, default: int = 0) -> int:
    try:
        return max(0, min(10, int(round(float(value)))))
    except Exception:
        return default


def _infer_reason_code(binary_checks: dict[str, bool], explicit: Any = None) -> QPReasonCode:
    if explicit:
        try:
            return QPReasonCode(explicit)
        except Exception:
            pass
    if binary_checks.get("source_alone_sufficient") is True:
        return QPReasonCode.QP_NOT_CIT_DEP
    if (
        binary_checks.get("target_relevant") is False
        or binary_checks.get("target_adds_essential_information") is False
    ):
        return QPReasonCode.QP_WRONG_TARGET
    if binary_checks.get("source_relevant") is False:
        return QPReasonCode.QP_SCOPE_MISMATCH
    if binary_checks.get("answer_supported") is False:
        return QPReasonCode.QP_ILL_FORMED
    if binary_checks.get("citation_dependent") is False:
        return QPReasonCode.QP_NOT_CIT_DEP
    return QPReasonCode.QP_ILL_FORMED


def judge_response_from_json(
    item_id: str,
    obj: dict[str, Any],
    *,
    pass_threshold: int = 7,
) -> JudgeResponse:
    """Parse v2 judge JSON while accepting legacy judge outputs."""
    binary_raw = obj.get("binary_checks") if isinstance(obj.get("binary_checks"), dict) else {}
    binary_checks = {
        "source_relevant": _as_bool(binary_raw.get("source_relevant")),
        "target_relevant": _as_bool(binary_raw.get("target_relevant")),
        "source_alone_sufficient": _as_bool(binary_raw.get("source_alone_sufficient")),
        "target_alone_sufficient": _as_bool(binary_raw.get("target_alone_sufficient")),
        "target_adds_essential_information": _as_bool(
            binary_raw.get("target_adds_essential_information")
        ),
        "answer_supported": _as_bool(binary_raw.get("answer_supported")),
        "citation_dependent": _as_bool(binary_raw.get("citation_dependent")),
    }
    binary_checks = {k: v for k, v in binary_checks.items() if v is not None}

    subscores_raw = obj.get("subscores") if isinstance(obj.get("subscores"), dict) else {}
    subscores = {
        key: _as_score(subscores_raw.get(key), default=0)
        for key in (
            "realism",
            "source_relevance",
            "target_relevance",
            "source_insufficiency",
            "target_necessity",
            "answer_support",
        )
        if key in subscores_raw
    }

    final_score = (
        _as_score(obj.get("final_score"), default=0)
        if obj.get("final_score") is not None
        else None
    )

    is_v2 = bool(obj.get("passed") is not None or binary_checks or final_score is not None)

    if is_v2:
        source_relevant = binary_checks.get("source_relevant") is True
        target_relevant = binary_checks.get("target_relevant") is True
        source_alone_sufficient = binary_checks.get("source_alone_sufficient") is True
        target_adds = binary_checks.get("target_adds_essential_information") is True
        answer_supported = binary_checks.get("answer_supported") is True
        citation_dependent = binary_checks.get("citation_dependent") is True
        score_ok = (final_score or 0) >= pass_threshold

        passed = bool(
            source_relevant
            and target_relevant
            and not source_alone_sufficient
            and target_adds
            and answer_supported
            and citation_dependent
            and score_ok
        )
        decision = QPDecision.PASS_QP if passed else QPDecision.DROP_QP
        reason = None if passed else _infer_reason_code(binary_checks, obj.get("reason_code_qp"))
        confidence = float(obj.get("confidence", (final_score or 0) / 10.0))
        source_alone_insufficient = not source_alone_sufficient
    else:
        if "decision_qp" not in obj or "confidence" not in obj:
            raise ValueError("Judge response missing both v2 fields and legacy decision_qp/confidence")
        decision = QPDecision(obj["decision_qp"])
        reason = QPReasonCode(obj["reason_code_qp"]) if obj.get("reason_code_qp") else None
        confidence = float(obj["confidence"])
        source_alone_insufficient = obj.get("source_alone_insufficient")
        source_alone_sufficient = (
            not source_alone_insufficient if isinstance(source_alone_insufficient, bool) else None
        )
        passed = decision == QPDecision.PASS_QP

    reasons = obj.get("reasons")
    if reasons is not None and not isinstance(reasons, list):
        reasons = [str(reasons)]

    return JudgeResponse(
        item_id=item_id,
        decision_qp=decision,
        reason_code_qp=reason,
        confidence=confidence,
        judge_schema_version=str(obj.get("judge_schema_version") or (JUDGE_SCHEMA_VERSION if is_v2 else "v1")),
        passed=passed,
        final_score=final_score,
        subscores=subscores or None,
        binary_checks=binary_checks or None,
        source_alone_insufficient=source_alone_insufficient,
        source_alone_sufficient=source_alone_sufficient,
        target_alone_sufficient=binary_checks.get("target_alone_sufficient"),
        target_adds_essential_information=binary_checks.get("target_adds_essential_information"),
        citation_dependent=binary_checks.get("citation_dependent"),
        answer_supported_by_judge=binary_checks.get("answer_supported"),
        target_contains_missing_detail=binary_checks.get("target_adds_essential_information"),
        key_missing_detail=obj.get("key_missing_detail"),
        reasons=reasons,
        support_snippets=obj.get("support_snippets"),
        notes=obj.get("notes"),
    )


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def aggregate_judge_passes(item_id: str, passes: list[JudgeResponse]) -> AggregatedJudgeResponse:
    if not passes:
        raise ValueError("Must have at least one pass to aggregate")

    n = len(passes)
    votes_pass = sum(1 for p in passes if p.decision_qp == QPDecision.PASS_QP)
    votes_drop = n - votes_pass
    vote_fraction = votes_pass / n

    score_pass = sum(p.confidence for p in passes if p.decision_qp == QPDecision.PASS_QP)
    score_drop = sum(p.confidence for p in passes if p.decision_qp == QPDecision.DROP_QP)
    total_score = score_pass + score_drop
    weighted_fraction = (max(score_pass, score_drop) / total_score) if total_score > 0 else 0.0

    # Final decision: weighted majority; tie -> conservative DROP
    if score_pass > score_drop:
        final_decision = QPDecision.PASS_QP
        final_reason = None
    elif score_drop > score_pass:
        final_decision = QPDecision.DROP_QP
        drop_reasons = [
            p.reason_code_qp
            for p in passes
            if p.decision_qp == QPDecision.DROP_QP and p.reason_code_qp
        ]
        final_reason = Counter(drop_reasons).most_common(1)[0][0] if drop_reasons else None
    else:
        final_decision = QPDecision.DROP_QP
        final_reason = None

    confidence_mean = sum(p.confidence for p in passes) / n

    # Flags: tune as needed, but keep conservative
    # Use majority_strength (max of votes_pass or votes_drop), not vote_fraction (which is only PASS)
    majority_strength = max(votes_pass, votes_drop) / n
    flag_low_consensus = (
        majority_strength < 0.67
        or weighted_fraction < 0.70
        or confidence_mean < 0.75  # Stricter: require higher confidence
    )

    def majority_bool(attr: str) -> bool | None:
        vals = [getattr(p, attr) for p in passes if getattr(p, attr) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v) > (len(vals) / 2)

    final_score_vals = [p.final_score for p in passes if p.final_score is not None]
    final_score = (
        int(round(sum(final_score_vals) / len(final_score_vals))) if final_score_vals else None
    )
    source_alone_sufficient = majority_bool("source_alone_sufficient")
    target_alone_sufficient = majority_bool("target_alone_sufficient")
    target_adds = majority_bool("target_adds_essential_information")
    citation_dependent = majority_bool("citation_dependent")
    answer_supported = majority_bool("answer_supported_by_judge")

    failure_flags = {
        "source_alone_sufficient": source_alone_sufficient is True,
        "target_not_necessary": target_adds is False,
        "unsupported_answer": answer_supported is False,
    }

    # V2 guardrail: even if legacy confidence voting would pass, explicit failed
    # binary checks force a conservative DROP.
    if (
        final_decision == QPDecision.PASS_QP
        and (
            source_alone_sufficient is True
            or target_adds is False
            or citation_dependent is False
            or answer_supported is False
        )
    ):
        final_decision = QPDecision.DROP_QP
        if source_alone_sufficient is True:
            final_reason = QPReasonCode.QP_NOT_CIT_DEP
        elif target_adds is False:
            final_reason = QPReasonCode.QP_WRONG_TARGET
        elif answer_supported is False:
            final_reason = QPReasonCode.QP_ILL_FORMED
        else:
            final_reason = QPReasonCode.QP_NOT_CIT_DEP

    runs = [
        {
            "judge_schema_version": p.judge_schema_version,
            "decision_qp": p.decision_qp.value,
            "reason_code_qp": p.reason_code_qp.value if p.reason_code_qp else None,
            "confidence": p.confidence,
            "passed": p.passed,
            "final_score": p.final_score,
            "subscores": p.subscores,
            "binary_checks": p.binary_checks,
            "reasons": p.reasons,
            "source_alone_sufficient": p.source_alone_sufficient,
            "target_alone_sufficient": p.target_alone_sufficient,
            "target_adds_essential_information": p.target_adds_essential_information,
            "citation_dependent": p.citation_dependent,
            "answer_supported_by_judge": p.answer_supported_by_judge,
        }
        for p in passes
    ]

    return AggregatedJudgeResponse(
        item_id=item_id,
        decision_qp_final=final_decision,
        reason_code_qp_final=final_reason,
        n_passes=n,
        votes_pass=votes_pass,
        votes_drop=votes_drop,
        confidence_mean=round(confidence_mean, 3),
        weighted_fraction=round(weighted_fraction, 3),
        flag_low_consensus=flag_low_consensus,
        judge_schema_version=JUDGE_SCHEMA_VERSION,
        passed=final_decision == QPDecision.PASS_QP,
        final_score=final_score,
        source_alone_sufficient=source_alone_sufficient,
        target_alone_sufficient=target_alone_sufficient,
        target_adds_essential_information=target_adds,
        citation_dependent=citation_dependent,
        answer_supported_by_judge=answer_supported,
        failure_flags=failure_flags,
        runs=runs,
    )


def process_judge_queue(
    queue: list[JudgeQueueItem],
    *,
    client: AzureOpenAI,
    deployment: str,
    num_passes: int = 1,
    temperature: float = 0.0,
    rate_limit_delay_s: float = 0.15,
    outer_retries: int = 2,
    max_completion_tokens: int = 700,
    pass_threshold: int = 7,
) -> list[AggregatedJudgeResponse]:
    """
    Multi-pass per item, then aggregate.

    Recommended configuration (cost/stability balance):
    - num_passes=1: Deterministic, cost-effective; use 2 if reason code instability observed
    - temperature=0.0: Fully deterministic LLM behavior for consistent gate
    - Aggregation: Confidence-weighted vote, conservative tie→DROP, flags low consensus for audit
    """
    rng = random.Random(13)
    aggregated: list[AggregatedJudgeResponse] = []

    total = len(queue)
    logger.info("Processing judge queue: %d items, %d pass(es) each", total, num_passes)

    for idx, item in enumerate(queue, 1):
        logger.info("Judging item %d/%d: %s", idx, total, item.item_id)
        passes: list[JudgeResponse] = []

        for pidx in range(num_passes):
            last_err: Exception | None = None
            for attempt in range(outer_retries + 1):
                try:
                    r = call_judge_llm_once(
                        client,
                        deployment=deployment,
                        queue_item=item,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        pass_threshold=pass_threshold,
                    )
                    passes.append(r)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < outer_retries:
                        time.sleep(0.3 + rng.random() * 0.6)
                    else:
                        logger.error(
                            "Judge pass failed: item=%s pass=%d err=%s", item.item_id, pidx + 1, e
                        )

            if last_err is not None:
                # Conservative fallback: error -> DROP_QP with QP_ILL_FORMED (no new enum)
                passes.append(
                    JudgeResponse(
                        item_id=item.item_id,
                        decision_qp=QPDecision.DROP_QP,
                        reason_code_qp=QPReasonCode.QP_ILL_FORMED,
                        confidence=0.0,
                        judge_schema_version=JUDGE_SCHEMA_VERSION,
                        passed=False,
                        final_score=0,
                        binary_checks={
                            "source_relevant": False,
                            "target_relevant": False,
                            "source_alone_sufficient": False,
                            "target_alone_sufficient": False,
                            "target_adds_essential_information": False,
                            "answer_supported": False,
                            "citation_dependent": False,
                        },
                        source_alone_sufficient=False,
                        target_alone_sufficient=False,
                        target_adds_essential_information=False,
                        citation_dependent=False,
                        answer_supported_by_judge=False,
                        reasons=["Judge execution error"],
                        notes=f"Judge error: {type(last_err).__name__}: {last_err}",
                    )
                )

            if rate_limit_delay_s > 0 and (pidx < num_passes - 1):
                time.sleep(rate_limit_delay_s)

        agg = aggregate_judge_passes(item.item_id, passes)
        if agg.flag_low_consensus:
            logger.warning(
                "Low consensus: item=%s votes_pass=%d/%d conf_mean=%.3f weighted=%.3f",
                item.item_id,
                agg.votes_pass,
                agg.n_passes,
                agg.confidence_mean,
                agg.weighted_fraction,
            )
        aggregated.append(agg)

    logger.info("Completed judge processing: %d aggregated responses", len(aggregated))
    return aggregated


def write_judge_output(
    queue: list[JudgeQueueItem],
    aggregated_responses: list[AggregatedJudgeResponse],
    output_dir: Path,
    *,
    model: str,
    temperature: float,
    input_file: Path,
    num_passes: int,
    pass_threshold: int,
) -> None:
    ensure_dir(output_dir)

    queue_file = output_dir / "judge_queue.jsonl"
    with open(queue_file, "w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    logger.info("Wrote judge queue: %s (n=%d)", queue_file, len(queue))

    out_file = output_dir / "judge_responses_aggregated.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for resp in aggregated_responses:
            d = asdict(resp)
            d["decision_qp_final"] = resp.decision_qp_final.value
            d["reason_code_qp_final"] = (
                resp.reason_code_qp_final.value if resp.reason_code_qp_final else None
            )
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Wrote aggregated responses: %s (n=%d)", out_file, len(aggregated_responses))

    response_by_id = {r.item_id: r for r in aggregated_responses}
    annotated_items_file = output_dir.parent / "curated_items.judge.with_judge_metadata.jsonl"
    if input_file.exists():
        with open(input_file, encoding="utf-8") as in_f, open(
            annotated_items_file, "w", encoding="utf-8"
        ) as out_f:
            for line in in_f:
                if not line.strip():
                    continue
                item = json.loads(line)
                resp = response_by_id.get(item.get("item_id"))
                if resp is not None:
                    item.update(
                        {
                            "judge_schema_version": resp.judge_schema_version,
                            "source_alone_sufficient": resp.source_alone_sufficient,
                            "target_alone_sufficient": resp.target_alone_sufficient,
                            "target_adds_essential_information": (
                                resp.target_adds_essential_information
                            ),
                            "citation_dependent": resp.citation_dependent,
                            "answer_supported_by_judge": resp.answer_supported_by_judge,
                        }
                    )
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Wrote judge-annotated curated items: %s", annotated_items_file)

    # Write separate files for PASS and DROP (strictly citation-dependent benchmark)
    pass_file = output_dir / "judge_responses_pass.jsonl"
    drop_file = output_dir / "judge_responses_drop.jsonl"

    pass_responses = [r for r in aggregated_responses if r.decision_qp_final == QPDecision.PASS_QP]
    drop_responses = [r for r in aggregated_responses if r.decision_qp_final == QPDecision.DROP_QP]

    def write_split_file(filepath: Path, responses: list[AggregatedJudgeResponse]) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for resp in responses:
                d = asdict(resp)
                d["decision_qp_final"] = resp.decision_qp_final.value
                d["reason_code_qp_final"] = (
                    resp.reason_code_qp_final.value if resp.reason_code_qp_final else None
                )
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    write_split_file(pass_file, pass_responses)
    logger.info("Wrote PASS (citation-dependent only): %s (n=%d)", pass_file, len(pass_responses))

    write_split_file(drop_file, drop_responses)
    logger.info("Wrote DROP (failed validation): %s (n=%d)", drop_file, len(drop_responses))

    pass_qp = sum(1 for r in aggregated_responses if r.decision_qp_final == QPDecision.PASS_QP)
    drop_qp = sum(1 for r in aggregated_responses if r.decision_qp_final == QPDecision.DROP_QP)
    low_consensus = sum(1 for r in aggregated_responses if r.flag_low_consensus)
    failed_source_alone = sum(
        1 for r in aggregated_responses if (r.failure_flags or {}).get("source_alone_sufficient")
    )
    failed_target_not_necessary = sum(
        1 for r in aggregated_responses if (r.failure_flags or {}).get("target_not_necessary")
    )
    failed_unsupported_answer = sum(
        1 for r in aggregated_responses if (r.failure_flags or {}).get("unsupported_answer")
    )

    avg_conf = (
        (sum(r.confidence_mean for r in aggregated_responses) / len(aggregated_responses))
        if aggregated_responses
        else 0.0
    )
    avg_weighted = (
        (sum(r.weighted_fraction for r in aggregated_responses) / len(aggregated_responses))
        if aggregated_responses
        else 0.0
    )

    reason_codes = [
        r.reason_code_qp_final.value for r in aggregated_responses if r.reason_code_qp_final
    ]
    reason_breakdown = dict(Counter(reason_codes))

    stats = {
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "total_items": len(aggregated_responses),
        "number_judged": len(aggregated_responses),
        "number_passed": pass_qp,
        "pass_qp_count": pass_qp,
        "drop_qp_count": drop_qp,
        "failed_source_alone_sufficiency_count": failed_source_alone,
        "failed_target_not_necessary_count": failed_target_not_necessary,
        "failed_unsupported_answer_count": failed_unsupported_answer,
        "low_consensus_count": low_consensus,
        "avg_confidence_mean": round(avg_conf, 3),
        "avg_weighted_fraction": round(avg_weighted, 3),
        "reason_code_breakdown": reason_breakdown,
        "judge_model": model,
        "judge_temperature": temperature,
        "num_judge_passes": num_passes,
        "judge_pass_threshold": pass_threshold,
        "input_file": str(input_file),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(output_dir / "judge_stats.json", stats)
    logger.info("Wrote judge stats")


def run_judge(cfg: RunConfig) -> None:
    logger.info("Starting QP judge evaluation (Azure-only, answer-agnostic)")
    t0 = time.time()

    curate_out = Path(cfg.paths.curate_output_dir or cfg.paths.output_dir)
    judge_out = curate_out / "curate_judge"
    ensure_dir(judge_out)

    items_file = curate_out / "curated_items.judge.jsonl"
    if not items_file.exists():
        logger.warning("No judge items found at %s", items_file)
        return

    items = load_judge_items(items_file)
    if not items:
        logger.info("No items to judge")
        return

    passages_file = Path(cfg.paths.output_dir) / "generator" / "passages_index.jsonl"
    passages: dict[str, dict[str, Any]] = {}
    if passages_file.exists():
        logger.info("Loading passages_index from: %s", passages_file)
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = p.get("passage_uid")
                if pid:
                    passages[pid] = p
        logger.info("Loaded %d passages from passages_index", len(passages))
    else:
        logger.warning(
            "passages_index not found at %s. Judge will evaluate with empty passages.",
            passages_file,
        )

    queue = build_judge_queue(items, passages)
    if not queue:
        logger.info("Judge queue is empty after filtering")
        return

    # Azure config from env (preferred). Strip whitespace/newlines (common in copy-pasted keys)
    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", "").strip())
    api_key = (os.getenv("AZURE_OPENAI_API_KEY", "") or "").strip()
    api_version = (os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview") or "").strip()
    deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "") or "").strip()

    if not azure_endpoint or not api_key or not deployment:
        raise RuntimeError(
            "Missing Azure env vars for judge. Required: "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_GPT52"
        )

    judge_cfg = getattr(cfg, "judge", None) or {}
    temperature = float(judge_cfg.get("temperature", 0.0))
    num_passes = int(judge_cfg.get("num_judge_passes", 1))
    rate_delay = float(judge_cfg.get("rate_limit_delay", 0.15))
    outer_retries = int(judge_cfg.get("outer_retries", 2))
    max_completion_tokens = int(judge_cfg.get("max_completion_tokens", 700))
    pass_threshold = int(judge_cfg.get("score_threshold", 7))

    # Diagnostic: log what Azure credentials were resolved
    logger.info(
        "Azure env resolved: endpoint=%s api_version=%s deployment=%s api_key_len=%d api_key_mask=%s",
        azure_endpoint,
        api_version,
        deployment,
        len(api_key or ""),
        _mask(api_key or ""),
    )

    client = build_azure_client(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout_s=float(judge_cfg.get("timeout_s", 60.0)),
        max_retries=int(judge_cfg.get("client_max_retries", 0)),
    )

    aggregated = process_judge_queue(
        queue,
        client=client,
        deployment=deployment,
        num_passes=num_passes,
        temperature=temperature,
        rate_limit_delay_s=rate_delay,
        outer_retries=outer_retries,
        max_completion_tokens=max_completion_tokens,
        pass_threshold=pass_threshold,
    )

    write_judge_output(
        queue,
        aggregated,
        judge_out,
        model=deployment,
        temperature=temperature,
        input_file=items_file,
        num_passes=num_passes,
        pass_threshold=pass_threshold,
    )

    logger.info("QP judge evaluation complete in %.2fs", time.time() - t0)
