"""
ObliQA-XRef Generator runner.

Reads adapter outputs:
- passage_corpus.jsonl
- crossref_resolved.cleaned.csv

Builds canonical Pair objects, applies filters/sampling, runs DPEL and/or SCHEMA,
and writes outputs under cfg.paths.output_dir.

Features:
- DPEL: Direct passage-enhanced lexical method
- SCHEMA: Schema extraction + Q&A generation from schema anchors
- Both: Merge DPEL and SCHEMA Q&As
- Comprehensive reporting with per-method statistics tracking
- Smart caching: Skip already-processed pairs to save LLM costs
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obliqaxref.config import RunConfig
from obliqaxref.generate.common.filters import PairFilterConfig, filter_pairs
from obliqaxref.generate.common.io import read_csv_dicts, read_jsonl, write_jsonl
from obliqaxref.generate.common.llm import build_client
from obliqaxref.generate.common.validate import detect_citation_leakage, validate_qa_item
from obliqaxref.generate.dpel.report import DPELRunReport
from obliqaxref.generate.schema.report import SchemaRunReport
from obliqaxref.generate.types import (
    Pair,
    Passage,
    QAItem,
    ReferenceType,
    make_pair_uid,
    to_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Overrides passed from CLI
# ---------------------------------------------------------------------
@dataclass
class GenerateOverrides:
    preset: str = "smoke"  # smoke | dev | paper | full
    dry_run: bool = False  # if True: no LLM calls, but do I/O + filtering + report
    method: str = "both"  # dpel | schema | both
    model: str = ""  # Azure: deployment name; empty string means use environment default (AZURE_OPENAI_DEPLOYMENT_GPT52)
    temperature: float = 0.2  # ✅ DEFAULT: 0.2 for both DPEL and SCHEMA
    seed: int = 13

    row_sample_n: int = 5  # sample N rows from CSV before building pairs
    row_sample_seed: int = 13

    max_pairs: int = 5  # cap final number of pairs
    max_q_per_pair: int = 1  # cap #questions per pair

    dedup: bool = True
    drop_title_targets: bool = True
    dual_anchors_mode: str = "always"  # always | freeform_only | off
    no_citations: bool = True   # forbid rule/section IDs in Q/A prose (evidence tags still required); default on
    no_citations_in_question: bool = True   # QUESTION-only citation guard; default on
    citation_leakage_action: str = "keep"   # keep | filter | separate
    sampling_mode: str | None = None        # None = use cfg.sampling.sampling_mode

    def __post_init__(self) -> None:
        valid_actions = {"keep", "filter", "separate"}
        if self.citation_leakage_action not in valid_actions:
            raise ValueError(
                f"citation_leakage_action must be one of {sorted(valid_actions)!r}, "
                f"got {self.citation_leakage_action!r}"
            )
        valid_modes = {"always", "freeform_only", "off"}
        if self.dual_anchors_mode not in valid_modes:
            raise ValueError(
                f"dual_anchors_mode must be one of {sorted(valid_modes)!r}, "
                f"got {self.dual_anchors_mode!r}"
            )
        if self.sampling_mode is not None:
            from obliqaxref.generate.common.sampling import VALID_MODES
            if self.sampling_mode not in VALID_MODES:
                raise ValueError(
                    f"sampling_mode must be one of {sorted(VALID_MODES)!r}, "
                    f"got {self.sampling_mode!r}"
                )


def _apply_preset(o: GenerateOverrides) -> GenerateOverrides:
    # dataclass default instance (used to detect "not overridden" values)
    defaults = GenerateOverrides()

    preset = (o.preset or "smoke").lower()

    if preset == "smoke":
        preset_vals = dict(
            row_sample_n=5,
            max_pairs=5,
            max_q_per_pair=1,
        )
    elif preset == "dev":
        preset_vals = dict(
            row_sample_n=50,
            max_pairs=50,
            max_q_per_pair=2,
        )
    elif preset == "paper":
        preset_vals = dict(
            row_sample_n=500,
            max_pairs=500,
            max_q_per_pair=2,
        )
    elif preset == "full":
        preset_vals = dict(
            row_sample_n=1000,  # Sample 1000 rows randomly
            max_pairs=0,  # 0 = no limit, process all pairs
            max_q_per_pair=2,
        )
    else:
        preset_vals = {}

    # Only apply preset values to fields that are still at default.
    for k, v in preset_vals.items():
        if hasattr(o, k) and getattr(o, k) == getattr(defaults, k):
            setattr(o, k, v)

    return o


# ... rest of helper functions ...


# =========================================================================
# Loading / building
# =========================================================================
def _passage_from_row(d: dict) -> Passage:
    return Passage(
        passage_uid=str(d.get("passage_uid") or ""),
        doc_id=str(d.get("doc_id") or ""),
        passage=str(d.get("passage") or ""),
        passage_id=d.get("passage_id"),
        eId=d.get("eId"),
        tag=d.get("tag"),
        source_tag=d.get("source_tag"),
        title=d.get("title"),
        heading_path=list(d.get("heading_path") or []),
        doc_url=d.get("doc_url"),
        passage_url=d.get("passage_url"),
        anchor_id=d.get("anchor_id"),
        anchor_ids=list(d.get("anchor_ids") or []),
        refs=list(d.get("refs") or []),
    )


def _build_pairs(rows: list[dict], passage_index: dict) -> list[Pair]:
    pairs: list[Pair] = []
    for r in rows:
        src_uid = str(r.get("SourceID") or "").strip()
        tgt_uid = str(r.get("TargetID") or "").strip()
        if not src_uid or not tgt_uid:
            continue

        src_passage = passage_index.get(src_uid)
        tgt_passage = passage_index.get(tgt_uid)
        if src_passage is None or tgt_passage is None:
            src_text = str(r.get("SourcePassage") or "")
            tgt_text = str(r.get("TargetPassage") or "")
            if not src_text or not tgt_text:
                continue
            src_doc = str(r.get("SourceDocumentID") or "")
            tgt_doc = str(r.get("TargetDocumentID") or "")
            ref_text = str(r.get("ReferenceText") or "")
            ref_type = ReferenceType.normalize(r.get("ReferenceType"))
            pair_uid = make_pair_uid(ref_type, ref_text, src_uid, tgt_uid)
            pairs.append(
                Pair(
                    pair_uid=pair_uid,
                    reference_type=ref_type,
                    reference_text=ref_text,
                    source_passage_uid=src_uid,
                    target_passage_uid=tgt_uid,
                    source_doc_id=src_doc,
                    target_doc_id=tgt_doc,
                    source_text=src_text,
                    target_text=tgt_text,
                    source_passage_id=r.get("SourcePassageID"),
                    target_passage_id=r.get("TargetPassageID"),
                )
            )
            continue

        ref_text = str(r.get("ReferenceText") or "")
        ref_type = ReferenceType.normalize(r.get("ReferenceType"))
        pair_uid = make_pair_uid(ref_type, ref_text, src_uid, tgt_uid)

        pairs.append(
            Pair(
                pair_uid=pair_uid,
                reference_type=ref_type,
                reference_text=ref_text,
                source_passage_uid=src_uid,
                target_passage_uid=tgt_uid,
                source_doc_id=src_passage.doc_id,
                target_doc_id=tgt_passage.doc_id,
                source_text=src_passage.text(),
                target_text=tgt_passage.text(),
                source_passage_id=src_passage.passage_id,
                target_passage_id=tgt_passage.passage_id,
                source_url=src_passage.passage_url,
                target_url=tgt_passage.passage_url,
                source_title=src_passage.title,
                target_title=tgt_passage.title,
                source_heading_path=list(src_passage.heading_path or []),
                target_heading_path=list(tgt_passage.heading_path or []),
            )
        )
    return pairs


def _ensure_dirs(work_dir: Path, out_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


# =========================================================================
# DPEL invocation (signature-robust)
# =========================================================================
def _call_generate_qas_for_pair(*, pair: Pair, client, o: GenerateOverrides):
    """
    Calls obliqaxref.generate.dpel.generate.generate_qas_for_pair.
    Returns DPELPairResult with qas list and drop counts.
    """
    from obliqaxref.generate.dpel.generate import DPELGenConfig, generate_qas_for_pair

    cfg = DPELGenConfig(
        model=o.model,
        temperature=o.temperature,
        seed=o.seed,
        max_q_per_pair=o.max_q_per_pair,
        no_citations=o.no_citations,
        no_citations_in_question=o.no_citations_in_question,
    )

    result = generate_qas_for_pair(client=client, pair=pair, cfg=cfg)
    return result


# =========================================================================
# CACHING HELPER (module-level, but called within run())
# =========================================================================
def _load_existing_qas(qa_path: str) -> set[str]:
    """Load pair_uids that have already been processed."""
    processed = set()
    if Path(qa_path).exists():
        try:
            with open(qa_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        pair_uid = obj.get("pair_uid")
                        if pair_uid:
                            processed.add(pair_uid)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Error loading existing QAs from %s: %s", qa_path, e)
    return processed


def _load_existing_records(records_path: str) -> dict[str, Any]:
    """Load existing SCHEMA extraction records by pair_uid."""
    records_by_uid = {}
    if Path(records_path).exists():
        try:
            for obj in read_jsonl(records_path):
                pair_uid = obj.get("pair_uid")
                if pair_uid:
                    records_by_uid[pair_uid] = obj
        except Exception as e:
            logger.warning("Error loading existing records from %s: %s", records_path, e)
    return records_by_uid


# ... rest of module-level functions ...


# =========================================================================
# LEAKAGE ANNOTATION HELPER
# =========================================================================
def _annotate_leakage(qa: QAItem, reference_text: str | None) -> QAItem:
    """Run detect_citation_leakage on qa.question and return an annotated copy."""
    result = detect_citation_leakage(qa.question, reference_text=reference_text)
    return dataclasses.replace(
        qa,
        citation_leakage=result["has_leakage"],
        citation_leakage_matches=result.get("matched_spans", []),
        citation_leakage_types=result.get("leakage_type", []),
    )


# =========================================================================
# Main entry
# =========================================================================
def run(cfg: RunConfig, o: GenerateOverrides | None = None) -> None:
    o = _apply_preset(o or GenerateOverrides())

    # -------------------------------------------------------------------------
    # PILOT MODE: override sampling parameters and redirect output directory
    # -------------------------------------------------------------------------
    pilot_cfg = cfg.pilot
    pilot_active = pilot_cfg.pilot_mode
    if pilot_active:
        logger.warning(
            "*** PILOT MODE ACTIVE: sampling %d xrefs, seed=%d, output suffix=%r ***",
            pilot_cfg.pilot_n_xrefs_per_corpus,
            pilot_cfg.pilot_random_seed,
            pilot_cfg.pilot_output_suffix,
        )
        o.row_sample_n = pilot_cfg.pilot_n_xrefs_per_corpus
        o.row_sample_seed = pilot_cfg.pilot_random_seed

    logger.info(
        "Generator starting. preset=%s method=%s dry_run=%s model=%s row_sample_n=%s max_pairs=%s max_q_per_pair=%s",
        o.preset,
        o.method,
        o.dry_run,
        o.model,
        o.row_sample_n,
        o.max_pairs,
        o.max_q_per_pair,
    )

    input_dir = Path(cfg.paths.input_dir)
    work_dir = Path(cfg.paths.work_dir)
    base_out_dir = Path(cfg.paths.output_dir)
    # Redirect output to pilot-specific directory when pilot mode is on
    out_dir = (
        base_out_dir.parent / f"{base_out_dir.name}_{pilot_cfg.pilot_output_suffix}"
        if pilot_active
        else base_out_dir
    )
    _ensure_dirs(work_dir, out_dir)

    # Create subdirectories for organized outputs
    dpel_dir = out_dir / "dpel"
    schema_dir = out_dir / "schema"
    stats_dir = out_dir / "stats"

    dpel_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Inputs from adapter
    corpus_path = input_dir / "passage_corpus.jsonl"
    xref_path = input_dir / "crossref_resolved.cleaned.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing: {corpus_path}")
    if not xref_path.exists():
        raise FileNotFoundError(f"Missing: {xref_path}")

    logger.info("Loading corpus: %s", corpus_path)
    corpus_rows = read_jsonl(str(corpus_path))
    passages = [_passage_from_row(r) for r in corpus_rows]
    passage_index = {p.passage_uid: p for p in passages if p.passage_uid}
    logger.info("Loaded passages: %d (index=%d)", len(passages), len(passage_index))

    logger.info("Loading crossrefs: %s", xref_path)
    xref_rows = read_csv_dicts(str(xref_path))
    logger.info("Loaded crossref rows: %d", len(xref_rows))

    # Sample rows using the configured difficulty-aware strategy
    from obliqaxref.generate.common.sampling import sample_xref_rows as _sample_xref_rows
    _sampling_mode = o.sampling_mode if o.sampling_mode is not None else cfg.sampling.sampling_mode
    rows, sampling_report_data = _sample_xref_rows(
        xref_rows,
        n=o.row_sample_n,
        mode=_sampling_mode,
        seed=o.row_sample_seed,
        max_jaccard_for_low_overlap=cfg.sampling.max_jaccard_for_low_overlap,
        mixed_difficulty_bucket_weights=cfg.sampling.mixed_difficulty_bucket_weights,
    )
    logger.info(
        "Row-sampled crossref rows: %d (mode=%s, n_requested=%d)",
        len(rows),
        _sampling_mode,
        o.row_sample_n,
    )

    # Write sampling report and enriched xref rows to stats dir
    sampling_report_path = stats_dir / "sampling_report.json"
    sampling_report_path.write_text(
        json.dumps(sampling_report_data, indent=2), encoding="utf-8"
    )
    logger.info("Wrote sampling report: %s", sampling_report_path)
    sampled_xrefs_path = stats_dir / "sampled_xrefs_with_features.jsonl"
    with open(sampled_xrefs_path, "w", encoding="utf-8") as _sf:
        for _r in rows:
            _sf.write(json.dumps(_r) + "\n")
    logger.info("Wrote sampled xrefs with features: %s (%d rows)", sampled_xrefs_path, len(rows))

    # Build pairs (join with corpus)
    pairs = _build_pairs(rows, passage_index)
    logger.info("Built pairs after join: %d", len(pairs))

    # Dedup
    if o.dedup:
        uniq = {}
        for p in pairs:
            uniq[p.pair_uid] = p
        pairs = list(uniq.values())
        logger.info("After dedup pairs: %d", len(pairs))
    else:
        logger.info("Deduplication disabled. Pairs count: %d", len(pairs))

    # Filters
    pf = PairFilterConfig(drop_title_targets=o.drop_title_targets)
    pairs, filter_report = filter_pairs(pairs, pf)
    logger.info("After filters pairs: %d", len(pairs))

    # Cap pairs
    if o.max_pairs and o.max_pairs > 0:
        pairs = pairs[: o.max_pairs]
    logger.info("Final pairs to process: %d", len(pairs))

    # Reference text lookup for leakage annotation (used by both DPEL and SCHEMA paths)
    # Define this upfront so SCHEMA-only runs (or when DPEL is skipped) don't hit
    # an UnboundLocalError when annotating citation leakage.
    pair_ref_lookup: dict[str, str | None] = {p.pair_uid: p.reference_text for p in pairs}

    # =========================================================================
    # CACHING: Load previously generated QAs to avoid re-running same pairs
    # =========================================================================
    dpel_qa_path = dpel_dir / "dpel.qa.jsonl"
    schema_qa_path = schema_dir / "schema.qa.jsonl"
    schema_extraction_path = schema_dir / "schema.extraction.jsonl"

    dpel_processed = (
        _load_existing_qas(str(dpel_qa_path)) if o.method in ("dpel", "both") else set()
    )
    schema_processed = (
        _load_existing_qas(str(schema_qa_path)) if o.method in ("schema", "both") else set()
    )
    schema_records_by_uid = (
        _load_existing_records(str(schema_extraction_path))
        if o.method in ("schema", "both")
        else {}
    )

    logger.info(
        "Resuming: DPEL has %d, SCHEMA has %d existing pairs, %d extraction records",
        len(dpel_processed),
        len(schema_processed),
        len(schema_records_by_uid),
    )

    # Filter pairs to skip already-processed ones
    pairs_to_process_dpel = (
        [p for p in pairs if p.pair_uid not in dpel_processed]
        if o.method in ("dpel", "both")
        else []
    )
    pairs_to_process_schema = (
        [p for p in pairs if p.pair_uid not in schema_processed]
        if o.method in ("schema", "both")
        else []
    )

    if len(pairs_to_process_dpel) < len(pairs):
        logger.info(
            "DPEL: Skipping %d already-processed pairs, running %d new pairs",
            len(dpel_processed),
            len(pairs_to_process_dpel),
        )

    if len(pairs_to_process_schema) < len(pairs):
        logger.info(
            "SCHEMA: Skipping %d already-processed pairs, running %d new pairs",
            len(schema_processed),
            len(pairs_to_process_schema),
        )

    # Initialize report objects for DPEL and SCHEMA
    dpel_report = DPELRunReport(
        rows_loaded=len(xref_rows),
        kept_candidates=len(pairs),
        model=o.model,
        temperature=o.temperature,
        seed=o.seed,
        no_citations=o.no_citations,
        dedup=o.dedup,
    )

    schema_report = SchemaRunReport(
        rows_loaded=len(xref_rows),
        kept_candidates=len(pairs),
        extract_model=o.model,
        gen_model=o.model,
        extract_temperature=o.temperature,
        gen_temperature=o.temperature,
        extract_seed=o.seed,
        gen_seed=o.seed,
        no_citations=o.no_citations,
        dual_anchors_mode=o.dual_anchors_mode,
    )

    # Base report dict
    report: dict[str, Any] = {
        "run_id": cfg.run_id,
        "preset": o.preset,
        "method": o.method,
        "dry_run": o.dry_run,
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "n_passages": len(passages),
        "n_xref_rows_loaded": len(xref_rows),
        "n_pairs_built": len(pairs),
        "sampling_mode": _sampling_mode,
        "sampling_report": sampling_report_data,
        "outputs": {},
    }

    # Dry run: just write report
    if o.dry_run:
        logger.info("Dry-run enabled: no LLM calls will be made.")
        report_path = stats_dir / "generate_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote: %s", report_path)
        return

    # LLM client — respect configured backend (e.g., 'azure' | 'openai')
    try:
        provider = getattr(cfg.generation, "llm_backend", None)
    except Exception:
        provider = None
    client, default_model = build_client(provider=provider)
    effective_model = o.model or default_model

    # =========================================================================
    # DPEL METHOD
    # =========================================================================
    qa_items: list[QAItem] = []
    if o.method in ("dpel", "both"):
        logger.info("Running DPEL generation over %d pair(s)", len(pairs_to_process_dpel))

        # Load existing QAs if resuming
        if dpel_qa_path.exists() and dpel_processed:
            try:
                existing_qas = read_jsonl(str(dpel_qa_path))
                qa_items = [QAItem(**q) for q in existing_qas]
                logger.info("Loaded %d existing DPEL QAs", len(qa_items))
            except Exception as e:
                logger.warning("Error loading existing DPEL QAs: %s", e)
                qa_items = []

        for i, pair in enumerate(pairs_to_process_dpel, start=1):
            logger.info("DPEL pair %d/%d uid=%s", i, len(pairs_to_process_dpel), pair.pair_uid)

            # Call DPEL generate and get DPELPairResult
            dpel_result = _call_generate_qas_for_pair(pair=pair, client=client, o=o)

            # Update DPEL report with per-pair metrics
            dpel_report.merge_pair_result(
                qas_created=len(dpel_result.qas),
                dropped_dupe_qs=dpel_result.dropped_dupe_qs,
                dropped_missing_tags=dpel_result.dropped_missing_tags,
                dropped_invalid=dpel_result.dropped_invalid,
                dropped_citations_policy=dpel_result.dropped_citations_policy,
                model_fail=dpel_result.model_fail,
            )
            dpel_report.pairs_processed += 1

            logger.info(
                "DPEL pair %s: generated=%d, dropped_dupe=%d, dropped_invalid=%d, dropped_missing_tags=%d, dropped_citations_policy=%d, model_fail=%s",
                pair.pair_uid,
                len(dpel_result.qas),
                dpel_result.dropped_dupe_qs,
                dpel_result.dropped_invalid,
                dpel_result.dropped_missing_tags,
                dpel_result.dropped_citations_policy,
                dpel_result.model_fail,
            )

            # Validate and collect QAs
            for qa in dpel_result.qas[: o.max_q_per_pair]:
                result = validate_qa_item(
                    qa,
                    no_citations=o.no_citations,
                    min_words=100,  # Lenient min for validation (prompt enforces 160-230)
                    max_words=500,  # Lenient max for validation (prompt enforces 160-230)
                )
                if result.ok:
                    qa_items.append(qa)
                else:
                    logger.warning("Dropped invalid QA (pair=%s): %s", pair.pair_uid, result.errors)

        # Annotate leakage on all collected DPEL QAs
        pair_ref_lookup: dict[str, str] = {p.pair_uid: p.reference_text for p in pairs}
        dpel_annotated_final: list[QAItem] = []
        dpel_citation_explicit: list[QAItem] = []
        dpel_leakage_count = 0
        for qa in qa_items:
            annotated_qa = _annotate_leakage(qa, reference_text=pair_ref_lookup.get(qa.pair_uid))
            if annotated_qa.citation_leakage:
                dpel_leakage_count += 1
            if annotated_qa.citation_leakage and o.citation_leakage_action in ("filter", "separate"):
                if o.citation_leakage_action == "separate":
                    dpel_citation_explicit.append(annotated_qa)
            else:
                dpel_annotated_final.append(annotated_qa)

        dpel_dropped_leakage = sum(
            1 for qa in dpel_annotated_final
            if False  # already excluded above; count = qa_items - annotated_final - explicit
        )
        # Recalculate counts cleanly
        dpel_dropped_leakage = (
            dpel_leakage_count - len(dpel_citation_explicit)
            if o.citation_leakage_action == "filter"
            else 0
        )
        dpel_separated_leakage = len(dpel_citation_explicit)

        dpel_report.merge_leakage_result(
            generated_total=len(qa_items),
            citation_leakage_count=dpel_leakage_count,
            dropped_citation_leakage=dpel_dropped_leakage,
            separated_citation_leakage=dpel_separated_leakage,
        )

        logger.info(
            "DPEL leakage annotation: total=%d, leakage=%d, action=%s, dropped=%d, separated=%d",
            len(qa_items),
            dpel_leakage_count,
            o.citation_leakage_action,
            dpel_dropped_leakage,
            dpel_separated_leakage,
        )

        # Write DPEL Q&As (append mode if resuming)
        write_jsonl(str(dpel_qa_path), [to_json(x) for x in dpel_annotated_final])
        report["outputs"]["dpel/dpel.qa.jsonl"] = str(dpel_qa_path)
        report["n_dpel_qas"] = len(dpel_annotated_final)
        report["dpel_stats"] = dpel_report.as_dict()
        logger.info("Wrote DPEL QA: %s (n=%d total)", dpel_qa_path, len(dpel_annotated_final))

        if dpel_citation_explicit:
            dpel_explicit_path = dpel_dir / "citation_explicit.qa.jsonl"
            write_jsonl(str(dpel_explicit_path), [to_json(x) for x in dpel_citation_explicit])
            report["outputs"]["dpel/citation_explicit.qa.jsonl"] = str(dpel_explicit_path)
            report["n_dpel_citation_explicit"] = len(dpel_citation_explicit)
            logger.info(
                "Wrote DPEL citation-explicit QAs: %s (n=%d)",
                dpel_explicit_path,
                len(dpel_citation_explicit),
            )

    # =========================================================================
    # SCHEMA METHOD
    # =========================================================================
    schema_records = []
    schema_qa_items: list[QAItem] = []
    if o.method in ("schema", "both"):
        logger.info(
            "Running SCHEMA extraction + generation over %d pair(s)", len(pairs_to_process_schema)
        )

        from obliqaxref.generate.schema.extract import (
            SchemaExtractConfig,
            extract_schema_for_pair,
            schema_pair_result_to_dict,
        )
        from obliqaxref.generate.schema.generate import (
            SchemaGenConfig,
            generate_qas_for_schema,
        )

        # Load existing records and QAs if resuming
        if schema_extraction_path.exists() and schema_records_by_uid:
            try:
                schema_records = read_jsonl(str(schema_extraction_path))
                logger.info("Loaded %d existing SCHEMA extraction records", len(schema_records))
            except Exception as e:
                logger.warning("Error loading existing SCHEMA records: %s", e)
                schema_records = []

        if schema_qa_path.exists() and schema_processed:
            try:
                existing_qas = read_jsonl(str(schema_qa_path))
                schema_qa_items = [QAItem(**q) for q in existing_qas]
                logger.info("Loaded %d existing SCHEMA QAs", len(schema_qa_items))
            except Exception as e:
                logger.warning("Error loading existing SCHEMA QAs: %s", e)
                schema_qa_items = []

        for i, pair in enumerate(pairs_to_process_schema, start=1):
            logger.info("SCHEMA pair %d/%d uid=%s", i, len(pairs_to_process_schema), pair.pair_uid)

            # ===== Step 1: Extract schema =====
            extract_cfg = SchemaExtractConfig(
                model=o.model,
                temperature=o.temperature,
                seed=o.seed,
                max_records_per_pair=o.max_q_per_pair,
                no_citations=o.no_citations,
            )
            schema_result = extract_schema_for_pair(client=client, pair=pair, cfg=extract_cfg)

            # Track extraction in report
            if schema_result.error:
                logger.warning(
                    "SCHEMA extraction error for pair %s: %s", pair.pair_uid, schema_result.error
                )
                schema_report.merge_extract_result(extracted=False, error=True)
                schema_records.append(schema_pair_result_to_dict(schema_result))
                continue

            schema_records.append(schema_pair_result_to_dict(schema_result))

            # Check if target is title (which causes skip)
            if schema_result.target_is_title:
                schema_report.merge_extract_result(extracted=False, dropped_title_targets=True)
            else:
                schema_report.merge_extract_result(extracted=True)

            schema_report.pairs_processed += 1

            logger.info(
                "SCHEMA extracted from pair %s: semantic_hook=%d chars, answer_spans=%d, target_is_title=%s",
                pair.pair_uid,
                len(schema_result.semantic_hook),
                len(schema_result.answer_spans),
                schema_result.target_is_title,
            )

            # ===== Step 2: Generate Q&As from schema =====
            gen_cfg = SchemaGenConfig(
                model=o.model,
                temperature=o.temperature,
                seed=o.seed,
                max_q_per_pair=o.max_q_per_pair,
                no_citations=o.no_citations,
                no_citations_in_question=o.no_citations_in_question,
                dual_anchors_mode=o.dual_anchors_mode,
            )
            qas, gen_meta = generate_qas_for_schema(
                client=client,
                schema_result=schema_result,
                source_text=pair.source_text,
                target_text=pair.target_text,
                cfg=gen_cfg,
            )

            # Track generation in report
            schema_report.merge_gen_result(
                qas_created=gen_meta["generated"],
                dropped_dupe_qs=gen_meta.get("dropped_dupe_qs", 0),
                dropped_missing_tags=gen_meta["dropped_missing_tags"],
                dropped_invalid=gen_meta["dropped_invalid"],
                model_fail=bool(gen_meta["error"]),
            )

            logger.info(
                "SCHEMA generated for pair %s: qas=%d, dropped_invalid=%d, dropped_missing_tags=%d, error=%s",
                pair.pair_uid,
                gen_meta["generated"],
                gen_meta["dropped_invalid"],
                gen_meta["dropped_missing_tags"],
                gen_meta["error"],
            )

            # Validate and add to collection
            for qa in qas:
                result = validate_qa_item(
                    qa,
                    no_citations=o.no_citations,
                    min_words=100,  # Lenient min for validation (prompt enforces 160-230)
                    max_words=500,  # Lenient max for validation (prompt enforces 160-230)
                )
                if result.ok:
                    schema_qa_items.append(qa)
                else:
                    logger.warning(
                        "Dropped invalid SCHEMA QA (pair=%s): %s", pair.pair_uid, result.errors
                    )

        # Write SCHEMA extraction records
        write_jsonl(str(schema_extraction_path), schema_records)
        report["outputs"]["schema/schema.extraction.jsonl"] = str(schema_extraction_path)
        report["n_schema_pairs"] = len(schema_records)
        report["n_schema_records"] = sum(len(r.get("answer_spans", [])) for r in schema_records)
        logger.info(
            "Wrote SCHEMA extraction records: %s (n_pairs=%d, n_records=%d)",
            schema_extraction_path,
            len(schema_records),
            report["n_schema_records"],
        )

        # Annotate leakage on all collected SCHEMA QAs
        schema_annotated_final: list[QAItem] = []
        schema_citation_explicit: list[QAItem] = []
        schema_leakage_count = 0
        for qa in schema_qa_items:
            annotated_qa = _annotate_leakage(qa, reference_text=pair_ref_lookup.get(qa.pair_uid))
            if annotated_qa.citation_leakage:
                schema_leakage_count += 1
            if annotated_qa.citation_leakage and o.citation_leakage_action in ("filter", "separate"):
                if o.citation_leakage_action == "separate":
                    schema_citation_explicit.append(annotated_qa)
            else:
                schema_annotated_final.append(annotated_qa)

        schema_dropped_leakage = (
            schema_leakage_count - len(schema_citation_explicit)
            if o.citation_leakage_action == "filter"
            else 0
        )
        schema_separated_leakage = len(schema_citation_explicit)

        schema_report.merge_leakage_result(
            generated_total=len(schema_qa_items),
            citation_leakage_count=schema_leakage_count,
            dropped_citation_leakage=schema_dropped_leakage,
            separated_citation_leakage=schema_separated_leakage,
        )

        logger.info(
            "SCHEMA leakage annotation: total=%d, leakage=%d, action=%s, dropped=%d, separated=%d",
            len(schema_qa_items),
            schema_leakage_count,
            o.citation_leakage_action,
            schema_dropped_leakage,
            schema_separated_leakage,
        )

        # Write SCHEMA Q&As
        write_jsonl(str(schema_qa_path), [to_json(x) for x in schema_annotated_final])
        report["outputs"]["schema/schema.qa.jsonl"] = str(schema_qa_path)
        report["n_schema_qas"] = len(schema_annotated_final)
        report["schema_stats"] = schema_report.as_dict()
        logger.info("Wrote SCHEMA QAs: %s (n=%d total)", schema_qa_path, len(schema_annotated_final))

        if schema_citation_explicit:
            schema_explicit_path = schema_dir / "citation_explicit.qa.jsonl"
            write_jsonl(str(schema_explicit_path), [to_json(x) for x in schema_citation_explicit])
            report["outputs"]["schema/citation_explicit.qa.jsonl"] = str(schema_explicit_path)
            report["n_schema_citation_explicit"] = len(schema_citation_explicit)
            logger.info(
                "Wrote SCHEMA citation-explicit QAs: %s (n=%d)",
                schema_explicit_path,
                len(schema_citation_explicit),
            )

        # If method=="both", merge schema QAs into qa_items
        if o.method == "both":
            qa_items.extend(schema_annotated_final)

    # =========================================================================
    # Final report
    # =========================================================================
    # Final report
    # =========================================================================
    report["pilot_mode"] = pilot_active
    if pilot_active:
        report["pilot_n_xrefs_per_corpus"] = pilot_cfg.pilot_n_xrefs_per_corpus
        report["pilot_random_seed"] = pilot_cfg.pilot_random_seed
        report["pilot_output_suffix"] = pilot_cfg.pilot_output_suffix

    report_path = stats_dir / "generate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote: %s", report_path)

    # Emit pilot report when pilot mode is active
    if pilot_active and not o.dry_run:
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir, generate_report=report)
        logger.info("Pilot report written to: %s", out_dir)


# =========================================================================
# CLI / __main__ entrypoint
# =========================================================================
def main(argv=None):
    """CLI entrypoint for run.py"""
    import argparse

    from obliqaxref.config import load_config

    ap = argparse.ArgumentParser(
        prog="python -m obliqaxref.generate.run",
        description="Run DPEL and/or SCHEMA generation with presets.",
    )

    ap.add_argument(
        "--preset",
        choices=["smoke", "dev", "paper", "full"],
        default="smoke",
        help="Preset configuration (smoke=5, dev=50, paper=500, full=unlimited pairs)",
    )
    ap.add_argument(
        "--method", choices=["dpel", "schema", "both"], default="both", help="Generation method(s)"
    )
    ap.add_argument("--model", default="gpt-4o-mini", help="Azure OpenAI deployment name")
    ap.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for LLM (default 0.2)"
    )
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument(
        "--max_q_per_pair", type=int, default=None, help="Override max questions per pair"
    )
    ap.add_argument("--no_citations", action="store_true", help="Disable citations in answers")
    ap.add_argument(
        "--dry_run", action="store_true", help="Dry run: scan/filter only, no LLM calls"
    )
    ap.add_argument(
        "--config", default=None, help="Path to config YAML (uses default if not provided)"
    )

    args = ap.parse_args(argv)

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        # Use default from environment
        cfg = load_config()

    # Build overrides
    overrides = GenerateOverrides(
        preset=args.preset,
        method=args.method,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_q_per_pair=args.max_q_per_pair or 2,
        dry_run=args.dry_run,
        no_citations=args.no_citations,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s - %(message)s")

    # Run
    run(cfg, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
