"""
Curation orchestration: Load IR runs, count votes, apply policy, output curated items.

This module:
1. Loads items, passages, and IR runs (TREC format)
2. Counts votes across IR retrievers using majority voting
3. Applies voting thresholds (KEEP/JUDGE/DROP)
4. Writes curated datasets split by decision
5. Computes and reports statistics
6. Calls judge LLM on JUDGE tier items for secondary validation
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obliqaxref.benchmark_metadata import (
    OBLIQA_XREF_METADATA_FIELDS,
    add_obliqa_xref_metadata_inplace,
)
from obliqaxref.config import RunConfig
from obliqaxref.utils.io import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class CurateOverrides:
    """CLI overrides for curate module."""

    preset: str = "dev"
    dry_run: bool = False
    skip_ir: bool = False
    skip_judge: bool = False
    skip_answer: bool = False
    ir_top_k: int | None = None
    keep_threshold: int | None = None
    judge_threshold: int | None = None
    judge_passes: int = 1
    judge_model: str = ""
    judge_temperature: float = 0.0


def load_items(items_file: Path) -> list[dict[str, Any]]:
    """Load items from JSONL file."""
    items = []
    with open(items_file) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_passages(passages_file: Path) -> dict[str, dict[str, Any]]:
    """Load passages from JSONL file and index by multiple possible ID keys.

    Index keys (if present): pid, passage_uid, passage_id, id
    The stored value is the original row dict.
    """
    passages: dict[str, dict[str, Any]] = {}
    try:
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                for k in ("pid", "passage_uid", "passage_id", "id"):
                    v = p.get(k)
                    if v:
                        passages[str(v)] = p
    except FileNotFoundError:
        pass
    return passages


def load_trec_runs(ir_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load IR runs from TREC format files."""
    runs = {}

    for trec_file in sorted(ir_dir.glob("*.trec")):
        run_name = trec_file.stem
        runs[run_name] = {}

        with open(trec_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[2]

                    if query_id not in runs[run_name]:
                        runs[run_name][query_id] = []
                    runs[run_name][query_id].append(doc_id)

    logger.info(f"  Loaded {len(runs)} IR runs: {list(runs.keys())}")
    return runs


def count_votes(
    item: dict[str, Any],
    runs: dict[str, dict[str, list[str]]],
) -> tuple[int, int]:
    """Count votes for source and target passages.

    Kept for backward compatibility. Prefer compute_detailed_votes() for new code.
    """
    item_id = item["item_id"]
    source_pid = item["source_passage_id"]
    target_pid = item["target_passage_id"]

    source_votes = 0
    target_votes = 0

    for run_name, results in runs.items():
        retrieved_pids = results.get(item_id, [])

        if source_pid in retrieved_pids:
            source_votes += 1
        if target_pid in retrieved_pids:
            target_votes += 1

    return source_votes, target_votes


def compute_detailed_votes(
    item: dict[str, Any],
    runs: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    """
    Compute per-retriever IR vote detail for a single item.

    Returns a dict with:
        source_vote_count          — number of retrievers that recovered the source passage
        target_vote_count          — number of retrievers that recovered the target passage
        both_vote_count            — number of retrievers that recovered BOTH passages
        retrievers_recovering_source  — list of run names that recovered source
        retrievers_recovering_target  — list of run names that recovered target
        retrievers_recovering_both    — list of run names that recovered both
    """
    item_id = item["item_id"]
    source_pid = item["source_passage_id"]
    target_pid = item["target_passage_id"]

    retrievers_recovering_source: list[str] = []
    retrievers_recovering_target: list[str] = []
    retrievers_recovering_both: list[str] = []

    for run_name, results in runs.items():
        retrieved_pids = results.get(item_id, [])
        got_source = source_pid in retrieved_pids
        got_target = target_pid in retrieved_pids
        if got_source:
            retrievers_recovering_source.append(run_name)
        if got_target:
            retrievers_recovering_target.append(run_name)
        if got_source and got_target:
            retrievers_recovering_both.append(run_name)

    return {
        "source_vote_count": len(retrievers_recovering_source),
        "target_vote_count": len(retrievers_recovering_target),
        "both_vote_count": len(retrievers_recovering_both),
        "retrievers_recovering_source": retrievers_recovering_source,
        "retrievers_recovering_target": retrievers_recovering_target,
        "retrievers_recovering_both": retrievers_recovering_both,
    }


def assign_ir_difficulty_label(
    source_vote_count: int,
    target_vote_count: int,
    both_vote_count: int,
    num_retrievers: int,
) -> str:
    """
    Assign a diagnostic IR difficulty label based on retriever vote counts.

    Labels (not used for selection — diagnostic only):
        easy        — majority (> 50 %) of retrievers recover both source and target
        medium      — at least one retriever recovers both, but not a majority
        hard        — no retriever recovers both, but ≥1 recovers each individually
        source_only — no retriever recovers both; source recovered, target not
        target_only — no retriever recovers both; target recovered, source not
        neither     — no retriever recovers source or target
    """
    if both_vote_count == 0:
        if source_vote_count > 0 and target_vote_count > 0:
            return "hard"
        if source_vote_count > 0:
            return "source_only"
        if target_vote_count > 0:
            return "target_only"
        return "neither"

    majority = (num_retrievers / 2) if num_retrievers > 0 else 0
    if both_vote_count > majority:
        return "easy"
    return "medium"


# ---------------------------------------------------------------------------
# Difficulty tier helpers
# ---------------------------------------------------------------------------

_CHALLENGING_LABELS: frozenset[str] = frozenset(
    {"hard", "source_only", "target_only", "neither"}
)


def assign_difficulty_tier(ir_difficulty_label: str) -> str:
    """Collapse 6-way IR difficulty label into a 2-way benchmark tier.

    Returns:
        ``"challenging"`` — IR systems failed to co-retrieve the evidence pair
            (labels: hard, source_only, target_only, neither).
        ``"retrievable"`` — majority or at least one retriever recovered both
            passages (labels: easy, medium).
    """
    return "challenging" if ir_difficulty_label in _CHALLENGING_LABELS else "retrievable"


# CSV columns written for every benchmark export
_BENCHMARK_CSV_FIELDS = (
    *OBLIQA_XREF_METADATA_FIELDS,
    "item_id",
    "question",
    "gold_answer",
    "source_passage_id",
    "target_passage_id",
    "source_text",
    "target_text",
    "reference_text",
    "reference_type",
    "method",
    "pair_uid",
    "persona",
    "citation_leakage",
    "citation_leakage_matches",
    "citation_leakage_types",
    "ir_difficulty_label",
    "difficulty_tier",
    "source_vote_count",
    "target_vote_count",
    "both_vote_count",
    "answer_validation_passed",
    "answer_validation_score",
    "answer_validation_reasons",
    "missing_source_tag",
    "missing_target_tag",
    "answer_responsive",
    "answer_supported",
    "judge_schema_version",
    "source_alone_sufficient",
    "target_alone_sufficient",
    "target_adds_essential_information",
    "citation_dependent",
    "answer_supported_by_judge",
)


def assemble_final_benchmark(
    curate_out: Path,
    items_file: Path,
    *,
    final_export_basis: str = "answer_valid",
) -> dict[str, Any]:
    """Assemble explicit final cohorts under the current curation policy.

    Reads:
        ``curate_judge/judge_responses_pass.jsonl`` — citation-dependency PASS ids
        ``curate_answer/answer_responses_pass.jsonl`` — answer-validation PASS ids, if present
        ``curate_answer/answer_responses_drop.jsonl`` — answer-validation DROP ids, if present
        ``curated_items.judge.jsonl`` — all items with IR metadata
        ``items_file`` — generator items (question / gold_answer text)

    Writes (inside *curate_out*):
        ``final_dependency_valid.jsonl/csv`` — judge PASS items
        ``final_answer_valid.jsonl/csv`` — judge PASS ∩ answer PASS
        ``final_answer_failed.jsonl/csv`` — judge PASS ∩ answer DROP
        ``final_benchmark.jsonl/csv`` — compatibility alias based on final_export_basis
        ``final_hard.jsonl/csv`` — challenging subset of the compatibility alias
        ``final_benchmark_stats.json`` — counts by difficulty_tier / label

    Returns a stats dict.
    """
    if final_export_basis not in {"dependency_valid", "answer_valid"}:
        raise ValueError("final_export_basis must be 'dependency_valid' or 'answer_valid'")

    judge_pass_file = curate_out / "curate_judge" / "judge_responses_pass.jsonl"
    answer_pass_file = curate_out / "curate_answer" / "answer_responses_pass.jsonl"
    answer_drop_file = curate_out / "curate_answer" / "answer_responses_drop.jsonl"
    answer_agg_file = curate_out / "curate_answer" / "answer_responses_aggregated.jsonl"
    judge_agg_file = curate_out / "curate_judge" / "judge_responses_aggregated.jsonl"
    ir_items_file = curate_out / "curated_items.judge.jsonl"

    # ---------------------------------------------------------------- load ids
    if not judge_pass_file.exists():
        logger.warning(
            "Judge PASS file not found (%s); skipping final benchmark assembly",
            judge_pass_file,
        )
        return {}

    judge_pass_ids: set[str] = set()
    judge_by_id: dict[str, dict[str, Any]] = {}
    with open(judge_pass_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("decision_qp_final") == "PASS_QP":
                iid = obj.get("item_id")
                if iid:
                    judge_pass_ids.add(iid)
                    judge_by_id[iid] = obj

    if judge_agg_file.exists():
        with open(judge_agg_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid:
                    judge_by_id[iid] = {**judge_by_id.get(iid, {}), **obj}

    answer_pass_ids: set[str] = set()
    answer_drop_ids: set[str] = set()
    answer_by_id: dict[str, dict[str, Any]] = {}
    if answer_pass_file.exists():
        with open(answer_pass_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("decision_ans_final") == "PASS_ANS":
                    iid = obj.get("item_id")
                    if iid:
                        answer_pass_ids.add(iid)
                        answer_by_id[iid] = obj
    else:
        logger.warning("Answer PASS file not found: %s", answer_pass_file)

    if answer_drop_file.exists():
        with open(answer_drop_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("decision_ans_final") == "DROP_ANS":
                    iid = obj.get("item_id")
                    if iid:
                        answer_drop_ids.add(iid)
                        answer_by_id[iid] = obj

    if answer_agg_file.exists():
        with open(answer_agg_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid:
                    answer_by_id[iid] = {**answer_by_id.get(iid, {}), **obj}

    dependency_valid_ids: set[str] = set(judge_pass_ids)
    answer_valid_ids: set[str] = judge_pass_ids & answer_pass_ids
    answer_failed_ids: set[str] = judge_pass_ids & answer_drop_ids
    logger.info(
        "  Final cohorts: dependency_valid=%d, answer_valid=%d, answer_failed=%d",
        len(judge_pass_ids),
        len(answer_valid_ids),
        len(answer_failed_ids),
    )

    # ---------------------------------------------------------- load metadata
    ir_items_by_id: dict[str, dict[str, Any]] = {}
    if ir_items_file.exists():
        with open(ir_items_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid:
                    ir_items_by_id[iid] = obj

    gen_items_by_id: dict[str, dict[str, Any]] = {}
    if items_file.exists():
        with open(items_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid:
                    gen_items_by_id[iid] = obj

    # ------------------------------------------ build final benchmark records
    all_candidate_ids = dependency_valid_ids | answer_valid_ids | answer_failed_ids

    def _reason_list(obj: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        reason = obj.get("reason_code_ans_final") or obj.get("reason_code_qp_final")
        if reason:
            reasons.append(str(reason))
        for run in obj.get("runs") or []:
            run_reason = run.get("reason_code_ans") or run.get("reason_code_qp")
            if run_reason and str(run_reason) not in reasons:
                reasons.append(str(run_reason))
            notes = run.get("notes")
            if notes and str(notes) not in reasons:
                reasons.append(str(notes))
        return reasons

    def _has_tag(answer: str, tag_prefix: str, pid: str) -> bool:
        return f"[#{tag_prefix}:{pid}]" in (answer or "")

    records_by_id: dict[str, dict[str, Any]] = {}
    for iid in sorted(all_candidate_ids):  # stable order
        gen = gen_items_by_id.get(iid, {})
        ir = ir_items_by_id.get(iid, {})
        judge = judge_by_id.get(iid, {})
        answer = answer_by_id.get(iid, {})

        ir_label = ir.get("ir_difficulty_label", "unknown")
        source_pid = gen.get("source_passage_id", ir.get("source_passage_id", ""))
        target_pid = gen.get("target_passage_id", ir.get("target_passage_id", ""))
        gold_answer = gen.get("gold_answer", ir.get("gold_answer", ""))
        answer_passed = iid in answer_valid_ids
        answer_failed = iid in answer_failed_ids
        record: dict[str, Any] = {
            "item_id": iid,
            "question": gen.get("question", ir.get("question", "")),
            "gold_answer": gold_answer,
            "source_passage_id": source_pid,
            "target_passage_id": target_pid,
            "source_text": gen.get("source_text", ir.get("source_text", "")),
            "target_text": gen.get("target_text", ir.get("target_text", "")),
            "reference_text": gen.get("reference_text", ir.get("reference_text", "")),
            "reference_type": gen.get("reference_type", ir.get("reference_type", "")),
            "method": gen.get("method", ir.get("method", "")),
            "pair_uid": gen.get("pair_uid", ir.get("pair_uid", "")),
            "persona": gen.get("persona", ir.get("persona", "")),
            "citation_leakage": gen.get("citation_leakage", ir.get("citation_leakage", False)),
            "citation_leakage_matches": gen.get(
                "citation_leakage_matches", ir.get("citation_leakage_matches", [])
            ),
            "citation_leakage_types": gen.get(
                "citation_leakage_types", ir.get("citation_leakage_types", [])
            ),
            "ir_difficulty_label": ir_label,
            "difficulty_tier": assign_difficulty_tier(ir_label),
            "source_vote_count": ir.get("source_vote_count", 0),
            "target_vote_count": ir.get("target_vote_count", 0),
            "both_vote_count": ir.get("both_vote_count", 0),
            "retrievers_recovering_source": ir.get("retrievers_recovering_source", []),
            "retrievers_recovering_target": ir.get("retrievers_recovering_target", []),
            "retrievers_recovering_both": ir.get("retrievers_recovering_both", []),
            "answer_validation_passed": answer_passed,
            "answer_validation_score": answer.get("confidence_mean"),
            "answer_validation_reasons": _reason_list(answer),
            "unsupported_claims": answer.get("unsupported_claims"),
            "missing_source_tag": not _has_tag(gold_answer, "SRC", source_pid),
            "missing_target_tag": not _has_tag(gold_answer, "TGT", target_pid),
            "answer_responsive": answer.get(
                "answer_responsive", answer.get("answer_addresses_question")
            ),
            "answer_supported": answer.get(
                "answer_supported", answer.get("answer_grounded_in_passages")
            ),
            "answer_validation_failed": answer_failed,
            "judge_schema_version": judge.get("judge_schema_version") or ir.get("judge_schema_version"),
            "source_alone_sufficient": judge.get(
                "source_alone_sufficient", ir.get("source_alone_sufficient")
            ),
            "target_alone_sufficient": judge.get(
                "target_alone_sufficient", ir.get("target_alone_sufficient")
            ),
            "target_adds_essential_information": judge.get(
                "target_adds_essential_information",
                ir.get("target_adds_essential_information"),
            ),
            "citation_dependent": judge.get("citation_dependent", ir.get("citation_dependent")),
            "answer_supported_by_judge": judge.get(
                "answer_supported_by_judge", ir.get("answer_supported_by_judge")
            ),
        }
        add_obliqa_xref_metadata_inplace(record)
        records_by_id[iid] = record

    dependency_valid_items = [records_by_id[iid] for iid in sorted(dependency_valid_ids)]
    answer_valid_items = [records_by_id[iid] for iid in sorted(answer_valid_ids)]
    answer_failed_items = [records_by_id[iid] for iid in sorted(answer_failed_ids)]

    compatibility_items = (
        dependency_valid_items if final_export_basis == "dependency_valid" else answer_valid_items
    )
    hard_items = [item for item in compatibility_items if item["difficulty_tier"] == "challenging"]

    # ------------------------------------------------------------- write JSONL
    def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _write_csv(path: Path, items: list[dict[str, Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(_BENCHMARK_CSV_FIELDS),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(items)

    _write_jsonl(curate_out / "final_dependency_valid.jsonl", dependency_valid_items)
    _write_csv(curate_out / "final_dependency_valid.csv", dependency_valid_items)
    _write_jsonl(curate_out / "final_answer_valid.jsonl", answer_valid_items)
    _write_csv(curate_out / "final_answer_valid.csv", answer_valid_items)
    _write_jsonl(curate_out / "final_answer_failed.jsonl", answer_failed_items)
    _write_csv(curate_out / "final_answer_failed.csv", answer_failed_items)

    _write_jsonl(curate_out / "final_benchmark.jsonl", compatibility_items)
    _write_csv(curate_out / "final_benchmark.csv", compatibility_items)
    _write_jsonl(curate_out / "final_hard.jsonl", hard_items)
    _write_csv(curate_out / "final_hard.csv", hard_items)

    logger.info(
        "  ✓ final_dependency_valid.jsonl / .csv  (%d items)", len(dependency_valid_items)
    )
    logger.info(
        "  ✓ final_answer_valid.jsonl / .csv      (%d items)", len(answer_valid_items)
    )
    logger.info(
        "  ✓ final_answer_failed.jsonl / .csv     (%d items)", len(answer_failed_items)
    )
    logger.info(
        "  ✓ final_benchmark.jsonl / .csv         (%d items; basis=%s)",
        len(compatibility_items),
        final_export_basis,
    )
    logger.info(
        "  ✓ final_hard.jsonl / .csv       (%d challenging items)", len(hard_items)
    )

    # ------------------------------------------------------------------ stats
    label_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {"retrievable": 0, "challenging": 0}
    for item in compatibility_items:
        lbl = item["ir_difficulty_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        tier_counts[item["difficulty_tier"]] = (
            tier_counts.get(item["difficulty_tier"], 0) + 1
        )

    stats: dict[str, Any] = {
        "generated_count": len(gen_items_by_id),
        "citation_dependency_passed_count": len(dependency_valid_items),
        "answer_validation_passed_count": len(answer_valid_items),
        "answer_validation_failed_count": len(answer_failed_items),
        "final_dependency_valid_count": len(dependency_valid_items),
        "final_answer_valid_count": len(answer_valid_items),
        "final_answer_failed_count": len(answer_failed_items),
        "final_export_basis": final_export_basis,
        "total_final": len(compatibility_items),
        "total_hard": len(hard_items),
        "difficulty_tier_counts": tier_counts,
        "ir_difficulty_label_counts": label_counts,
    }

    with open(curate_out / "final_benchmark_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def run(cfg: RunConfig, overrides: CurateOverrides | None = None) -> dict[str, Any]:
    """
    Run curation pipeline.

    Args:
        cfg: RunConfig from YAML
        overrides: CLI overrides (preset, dry_run, skip_ir, skip_judge, etc.)

    Workflow:
    1. Merge method-specific items if needed
    2. Load items, passages, and IR runs
    3. Count votes for each item
    4. Apply voting policy
    5. Separate items by decision tier
    6. Write outputs and statistics
    """

    start_time = time.time()

    # -------------------------------------------------------------------------
    # PILOT MODE: redirect I/O directories to pilot-specific paths
    # -------------------------------------------------------------------------
    pilot_cfg = cfg.pilot
    pilot_active = pilot_cfg.pilot_mode
    if pilot_active:
        suffix = pilot_cfg.pilot_output_suffix
        base_out = Path(cfg.paths.output_dir)
        pilot_out = base_out.parent / f"{base_out.name}_{suffix}"
        base_curate_out = Path(cfg.paths.curate_output_dir or cfg.paths.output_dir)
        pilot_curate_out = base_curate_out.parent / f"{base_curate_out.name}_{suffix}"
        patched_paths = cfg.paths.model_copy(
            update={
                "output_dir": str(pilot_out),
                "curate_output_dir": str(pilot_curate_out),
            }
        )
        cfg = cfg.model_copy(update={"paths": patched_paths})
        logger.warning(
            "*** PILOT MODE ACTIVE: curation reading from %s, writing to %s ***",
            pilot_out,
            pilot_curate_out,
        )

    output_path = cfg.paths.curate_output_dir or cfg.paths.output_dir
    out_dir = ensure_dir(output_path)

    # Apply CLI overrides
    if overrides:
        if overrides.keep_threshold is not None:
            cfg.curation.ir_agreement.keep_threshold = overrides.keep_threshold
        if overrides.judge_threshold is not None:
            cfg.curation.ir_agreement.judge_threshold = overrides.judge_threshold
        if overrides.ir_top_k is not None:
            cfg.curation.ir_agreement.top_k = overrides.ir_top_k

    logger.info("=" * 70)
    logger.info("CURATION: VOTING & FILTERING")
    logger.info("=" * 70)

    # Phase 1: IR Retrieval (optional)
    if overrides and not overrides.skip_ir:
        logger.info("\n[Phase 1] Running IR Retrieval...")
        try:
            from obliqaxref.curate.ir_retrieval import run_ir_retrieval

            run_ir_retrieval(cfg)
            logger.info("  ✓ IR retrieval complete")
        except Exception as e:
            logger.warning(f"IR retrieval failed: {e}. Using existing TREC runs.")
    elif overrides and overrides.skip_ir:
        logger.info("\n[Phase 1] Skipping IR retrieval (--skip-ir)")

    # Merge items if needed
    logger.info("\n[Phase 2] Preparing items...")

    # Look for items.jsonl in multiple possible locations
    items_path = None
    possible_paths = [
        Path(cfg.paths.input_dir) / "generator" / "items.jsonl",
        Path(cfg.paths.output_dir) / "generator" / "items.jsonl",
    ]

    for possible_path in possible_paths:
        if possible_path.exists():
            items_path = possible_path
            break

    if items_path is None:
        logger.info("  items.jsonl not found. Merging method-specific items...")
        from obliqaxref.curate.merge import merge_qa_items

        merge_qa_items(cfg.paths.output_dir, input_dir=Path(cfg.paths.input_dir))
        items_path = Path(cfg.paths.output_dir) / "generator" / "items.jsonl"
    else:
        logger.info(f"  ✓ items.jsonl found at {items_path}")

    # Load data
    logger.info("\n[Phase 2.1] Loading data...")

    passages_path = Path(cfg.paths.input_dir) / "passage_corpus.jsonl"
    ir_dir = Path(
        cfg.paths.output_dir
    )  # IR runs are written to the output directory by IR retrieval

    items = load_items(items_path)
    passages = load_passages(passages_path)
    runs = load_trec_runs(ir_dir)

    logger.info(f"  ✓ Loaded {len(items)} items, {len(passages)} passages")

    # Count votes and assign IR difficulty labels
    logger.info("\n[Phase 2.2] Computing IR votes and assigning difficulty labels...")

    num_retrievers = len(runs)
    all_items_annotated: list[dict[str, Any]] = []

    label_counts: dict[str, int] = {
        "easy": 0, "medium": 0, "hard": 0,
        "source_only": 0, "target_only": 0, "neither": 0,
    }

    for item in items:
        vote_detail = compute_detailed_votes(item, runs)
        difficulty_label = assign_ir_difficulty_label(
            vote_detail["source_vote_count"],
            vote_detail["target_vote_count"],
            vote_detail["both_vote_count"],
            num_retrievers,
        )
        label_counts[difficulty_label] = label_counts.get(difficulty_label, 0) + 1

        curated_item = dict(item)

        # IR metadata — stored as diagnostic; NOT used for selection
        curated_item["ir_difficulty_label"] = difficulty_label
        curated_item["source_vote_count"] = vote_detail["source_vote_count"]
        curated_item["target_vote_count"] = vote_detail["target_vote_count"]
        curated_item["both_vote_count"] = vote_detail["both_vote_count"]
        curated_item["retrievers_recovering_source"] = vote_detail["retrievers_recovering_source"]
        curated_item["retrievers_recovering_target"] = vote_detail["retrievers_recovering_target"]
        curated_item["retrievers_recovering_both"] = vote_detail["retrievers_recovering_both"]

        # All items proceed to citation-dependency judge regardless of IR outcome
        curated_item["decision_ir"] = "JUDGE_IR"

        # Attach passage text if available (support multiple field names)
        def _get_text(p: dict[str, Any] | None) -> str:
            if not p:
                return ""
            return (
                str(p.get("text") or p.get("passage") or p.get("content") or "").strip()
            )

        curated_item["source_text"] = _get_text(passages.get(item["source_passage_id"]))
        curated_item["target_text"] = _get_text(passages.get(item["target_passage_id"]))

        all_items_annotated.append(curated_item)

    total_items = len(all_items_annotated)
    logger.info("  ✓ IR difficulty labels assigned (%d items):", total_items)
    for lbl, cnt in sorted(label_counts.items()):
        pct = 100 * cnt / total_items if total_items else 0
        logger.info("      %-12s %4d  (%5.1f%%)", lbl, cnt, pct)

    # Write outputs
    logger.info("\n[Step 3] Writing outputs...")

    # All items go to the judge input file (selection is now purely validity-based)
    judge_file = out_dir / "curated_items.judge.jsonl"
    with open(judge_file, "w") as f:
        for item in all_items_annotated:
            f.write(json.dumps(item) + "\n")
    logger.info("  ✓ %s (%d items — all proceed to citation-dependency judge)", judge_file.name, total_items)

    # Decisions file (IR annotation summary — diagnostic)
    decisions = [
        {
            "item_id": item["item_id"],
            "ir_difficulty_label": item["ir_difficulty_label"],
            "source_vote_count": item["source_vote_count"],
            "target_vote_count": item["target_vote_count"],
            "both_vote_count": item["both_vote_count"],
            "source_passage_id": item["source_passage_id"],
            "target_passage_id": item["target_passage_id"],
        }
        for item in all_items_annotated
    ]
    decisions_file = out_dir / "decisions.jsonl"
    with open(decisions_file, "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    logger.info("  ✓ %s (IR annotation — diagnostic)", decisions_file.name)

    # Compute statistics
    logger.info("\n[Phase 2.3] Computing statistics...")

    stats = {
        "total_items": total_items,
        "ir_difficulty_label_counts": label_counts,
        "policy": {
            "ir_role": "diagnostic_only",
            "selection_criterion": "citation_dependency_validity + answer_validity",
            "note": (
                "IR agreement is stored as ir_difficulty_label metadata. "
                "Final benchmark selection is based on LLM citation-dependency judge "
                "and answer validation only."
            ),
        },
        "ir_runs": list(runs.keys()),
        "num_ir_methods": num_retrievers,
    }

    stats_file = out_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  ✓ {stats_file.name}")

    # Summary of IR annotation phase
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 (IR ANNOTATION) COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs written to: {out_dir}")

    # Phase 3: Citation-dependency judge on ALL items
    if total_items > 0 and not (overrides and overrides.skip_judge):
        logger.info("\n[Phase 3] Running citation-dependency judge on all %d items...", total_items)
        try:
            from obliqaxref.curate.judge import run_judge

            run_judge(cfg)
            logger.info("  ✓ Judge evaluation complete")
        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}. Continuing without judge filter.")
    elif overrides and overrides.skip_judge:
        logger.info("\n[Phase 3] Skipping judge (--skip-judge)")
    else:
        logger.info("\n[Phase 3] No items to judge")

    # Phase 4: Answer validation on judge PASS items
    if not (overrides and overrides.skip_answer):
        logger.info("\n[Phase 4] Running answer validation on citation-dependency PASS items...")
        try:
            from obliqaxref.curate.answer.run import run_answer_validation

            run_answer_validation(cfg)
            logger.info("  ✓ Answer validation complete")
            # Optionally surface answer stats into main stats
            answer_stats_file = (
                Path(cfg.paths.curate_output_dir or cfg.paths.output_dir)
                / "curate_answer"
                / "answer_stats.json"
            )
            if answer_stats_file.exists():
                try:
                    with open(answer_stats_file, encoding="utf-8") as f:
                        ans_stats = json.load(f)
                    stats["answer_validation"] = ans_stats
                    logger.info(
                        "  ✓ Answer stats: pass=%s drop=%s low_consensus=%s",
                        ans_stats.get("pass_ans_count"),
                        ans_stats.get("drop_ans_count"),
                        ans_stats.get("low_consensus_count"),
                    )
                except Exception:
                    logger.warning("Could not read answer_stats.json for summary")
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}. Continuing without answer filter.")
    else:
        logger.info("\n[Phase 4] Skipping answer validation (--skip-answer)")

    # Phase 5: Assemble final benchmark (runs unconditionally — skipped if
    # required upstream files don't exist yet, e.g. when phases 3/4 were skipped)
    logger.info("\n[Phase 5] Assembling final benchmark...")
    _items_file = Path(cfg.paths.output_dir) / "generator" / "items.jsonl"
    try:
        bench_stats = assemble_final_benchmark(
            out_dir,
            _items_file,
            final_export_basis=cfg.curation.final_export_basis,
        )
        if bench_stats:
            stats["final_benchmark"] = bench_stats
            logger.info(
                "  ✓ Final benchmark: %d items total (%d challenging)",
                bench_stats.get("total_final", 0),
                bench_stats.get("total_hard", 0),
            )
    except Exception as e:
        logger.warning("Final benchmark assembly failed: %s", e)

    # Final summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("CURATION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs written to: {out_dir}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 70)

    # Emit pilot report when pilot mode is active
    if pilot_active and not (overrides and overrides.dry_run):
        try:
            from obliqaxref.pilot.report import generate_pilot_report

            generate_pilot_report(out_dir=out_dir, curate_stats=stats)
            logger.info("Pilot report written to: %s", out_dir)
        except Exception as e:
            logger.warning("Pilot report generation failed: %s", e)

    return stats
