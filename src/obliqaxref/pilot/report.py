# src/obliqaxref/pilot/report.py
"""
Pilot run report generator.

Produces three files in the given output directory:
  - pilot_report.json     machine-readable summary
  - pilot_report.md       human-readable Markdown summary
  - pilot_examples.jsonl  example items (leakage failures, valid, hard, failed)

This module is intentionally stateless: it reads from files already written by
the generate and curate stages rather than accepting in-memory data, so it can
be called independently (e.g. to re-generate the report after a run).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How many examples to collect for each category
_MAX_EXAMPLES_PER_CATEGORY = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Silently return empty list if file is absent or unreadable."""
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return items


def _read_json(path: Path) -> dict[str, Any]:
    """Silently return empty dict if file is absent or unreadable."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * count / total:.1f}%"


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------

def _build_report(
    out_dir: Path,
    generate_report: dict[str, Any] | None = None,
    curate_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Collect metrics from on-disk files (and optionally in-memory dicts passed
    directly from generate/curate run.py) and return a structured report dict.
    """
    report: dict[str, Any] = {"pilot_mode": True, "output_dir": str(out_dir)}

    # ------------------------------------------------------------------
    # 1. Generation stats
    # ------------------------------------------------------------------
    gen_report = generate_report or _read_json(out_dir / "stats" / "generate_report.json")

    n_xrefs_sampled = gen_report.get("pilot_n_xrefs_per_corpus", gen_report.get("n_xref_rows_loaded", 0))
    report["n_xrefs_sampled"] = n_xrefs_sampled

    n_dpel = gen_report.get("n_dpel_qas", 0)
    n_schema = gen_report.get("n_schema_qas", 0)
    n_qas_total = n_dpel + n_schema
    report["n_qas_generated"] = {"dpel": n_dpel, "schema": n_schema, "total": n_qas_total}

    # Citation leakage rate (aggregate over both methods)
    total_leakage = 0
    total_gen_for_leakage = 0
    for method_key in ("dpel_stats", "schema_stats"):
        mstats = gen_report.get(method_key) or {}
        leakage = mstats.get("citation_leakage_count", 0) or 0
        gen_total = mstats.get("generated_total") or mstats.get("qas_created", 0) or 0
        total_leakage += leakage
        total_gen_for_leakage += gen_total

    leakage_rate = total_leakage / total_gen_for_leakage if total_gen_for_leakage > 0 else None
    report["citation_leakage"] = {
        "count": total_leakage,
        "total_checked": total_gen_for_leakage,
        "rate": round(leakage_rate, 4) if leakage_rate is not None else None,
        "rate_pct": _pct(total_leakage, total_gen_for_leakage),
    }

    # ------------------------------------------------------------------
    # 2. Curation / judge stats (if available)
    # ------------------------------------------------------------------
    cur_stats = curate_stats or _read_json(out_dir / "stats.json")
    report["curation"] = {}

    if cur_stats:
        ir_dist = cur_stats.get("ir_difficulty_label_counts") or {}
        report["curation"]["ir_difficulty_label_distribution"] = ir_dist

        final_bench = cur_stats.get("final_benchmark") or {}
        if final_bench:
            report["curation"]["final_benchmark"] = {
                "total_final": final_bench.get("total_final", 0),
                "total_hard": final_bench.get("total_hard", 0),
            }

        ans_val = cur_stats.get("answer_validation") or {}
        if ans_val:
            n_pass_ans = ans_val.get("pass_ans_count", 0) or 0
            n_total_ans = (n_pass_ans
                          + (ans_val.get("drop_ans_count") or 0)
                          + (ans_val.get("low_consensus_count") or 0))
            report["curation"]["answer_validation"] = {
                "pass": n_pass_ans,
                "total": n_total_ans,
                "pass_rate": round(n_pass_ans / n_total_ans, 4) if n_total_ans else None,
                "pass_rate_pct": _pct(n_pass_ans, n_total_ans),
            }

    # Judge pass rate — from judge_responses_pass.jsonl vs curated_items.judge.jsonl
    judge_pass_items = _read_jsonl(out_dir / "judge_responses_pass.jsonl")
    judge_all_items = _read_jsonl(out_dir / "curated_items.judge.jsonl")
    n_judge_pass = len(judge_pass_items)
    n_judge_total = len(judge_all_items) or None  # None means not available
    report["curation"]["judge"] = {
        "pass": n_judge_pass,
        "total": n_judge_total,
        "pass_rate_pct": _pct(n_judge_pass, n_judge_total) if n_judge_total else "N/A",
    }

    # ------------------------------------------------------------------
    # 3. Examples
    # ------------------------------------------------------------------
    examples: dict[str, list[dict[str, Any]]] = {
        "citation_leakage_failures": [],
        "valid_items": [],
        "hard_items": [],
        "failed_items": [],
    }

    # Pull examples from DPEL + SCHEMA qa files
    for method_dir, qa_file in [
        (out_dir / "dpel", "dpel.qa.jsonl"),
        (out_dir / "schema", "schema.qa.jsonl"),
    ]:
        all_qas = _read_jsonl(method_dir / qa_file)
        for qa in all_qas:
            if (
                qa.get("citation_leakage")
                and len(examples["citation_leakage_failures"]) < _MAX_EXAMPLES_PER_CATEGORY
            ):
                examples["citation_leakage_failures"].append({
                    "qa_uid": qa.get("qa_uid"),
                    "method": qa.get("method"),
                    "question": qa.get("question"),
                    "citation_leakage_matches": qa.get("citation_leakage_matches"),
                    "citation_leakage_types": qa.get("citation_leakage_types"),
                })
            elif (
                not qa.get("citation_leakage")
                and len(examples["valid_items"]) < _MAX_EXAMPLES_PER_CATEGORY
            ):
                examples["valid_items"].append({
                    "qa_uid": qa.get("qa_uid"),
                    "method": qa.get("method"),
                    "question": qa.get("question"),
                    "expected_answer": qa.get("expected_answer"),
                })

    # Hard items from final_hard.jsonl
    hard_items = _read_jsonl(out_dir / "final_hard.jsonl")
    for item in hard_items[:_MAX_EXAMPLES_PER_CATEGORY]:
        examples["hard_items"].append({
            "item_id": item.get("item_id"),
            "question": item.get("question"),
            "ir_difficulty_label": item.get("ir_difficulty_label"),
            "difficulty_tier": item.get("difficulty_tier"),
        })

    # Failed items — items in judge input that are NOT in judge pass (first N)
    judge_pass_ids = {it.get("item_id") for it in judge_pass_items}
    for item in judge_all_items:
        if (
            item.get("item_id") not in judge_pass_ids
            and len(examples["failed_items"]) < _MAX_EXAMPLES_PER_CATEGORY
        ):
            examples["failed_items"].append({
                "item_id": item.get("item_id"),
                "question": item.get("question"),
                "ir_difficulty_label": item.get("ir_difficulty_label"),
                "reason": "did_not_pass_citation_dependency_judge",
            })

    report["examples"] = examples
    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Pilot Run Report")
    lines.append("")
    lines.append(f"**Output directory:** `{report.get('output_dir', 'N/A')}`")
    lines.append("")

    # --- Generation ---
    lines.append("## Generation")
    lines.append("")
    lines.append(f"- **Sampled xrefs:** {report.get('n_xrefs_sampled', 'N/A')}")
    qas = report.get("n_qas_generated", {})
    lines.append(
        f"- **Generated QAs:** {qas.get('total', 0)} "
        f"(DPEL: {qas.get('dpel', 0)}, SCHEMA: {qas.get('schema', 0)})"
    )
    lk = report.get("citation_leakage", {})
    lines.append(
        f"- **Citation leakage rate:** {lk.get('rate_pct', 'N/A')} "
        f"({lk.get('count', 0)} / {lk.get('total_checked', 0)})"
    )
    lines.append("")

    # --- Curation ---
    cur = report.get("curation", {})
    lines.append("## Curation")
    lines.append("")
    judge = cur.get("judge", {})
    lines.append(
        f"- **Validation pass rate:** {judge.get('pass_rate_pct', 'N/A')} "
        f"({judge.get('pass', 0)} / {judge.get('total', 'N/A')})"
    )
    av = cur.get("answer_validation")
    if av:
        lines.append(
            f"- **Answer validation pass rate:** {av.get('pass_rate_pct', 'N/A')} "
            f"({av.get('pass', 0)} / {av.get('total', 'N/A')})"
        )
    ir_dist = cur.get("ir_difficulty_label_distribution")
    if ir_dist:
        lines.append("")
        lines.append("### IR Difficulty Label Distribution")
        lines.append("")
        lines.append("| Label | Count |")
        lines.append("|-------|-------|")
        for label, count in sorted(ir_dist.items()):
            lines.append(f"| {label} | {count} |")
    fb = cur.get("final_benchmark")
    if fb:
        lines.append("")
        lines.append(
            f"- **Final benchmark items:** {fb.get('total_final', 0)} "
            f"({fb.get('total_hard', 0)} challenging)"
        )
    lines.append("")

    # --- Examples ---
    examples = report.get("examples", {})

    def _section(title: str, items: list[dict[str, Any]], fields: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("_No examples available._")
            lines.append("")
            return
        for ex in items:
            lines.append("---")
            for field in fields:
                val = ex.get(field)
                if val is not None:
                    lines.append(f"**{field}:** {val}")
            lines.append("")

    _section(
        "Citation Leakage Failures",
        examples.get("citation_leakage_failures", []),
        ["qa_uid", "method", "question", "citation_leakage_matches", "citation_leakage_types"],
    )
    _section(
        "Valid Items (sample)",
        examples.get("valid_items", []),
        ["qa_uid", "method", "question", "expected_answer"],
    )
    _section(
        "Hard Items (sample)",
        examples.get("hard_items", []),
        ["item_id", "question", "ir_difficulty_label", "difficulty_tier"],
    )
    _section(
        "Failed Items (sample)",
        examples.get("failed_items", []),
        ["item_id", "question", "ir_difficulty_label", "reason"],
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pilot_report(
    out_dir: Path,
    *,
    generate_report: dict[str, Any] | None = None,
    curate_stats: dict[str, Any] | None = None,
) -> None:
    """
    Write pilot_report.json, pilot_report.md, and pilot_examples.jsonl to *out_dir*.

    Parameters
    ----------
    out_dir:
        The pilot output directory (e.g. ``runs/generate_adgm_pilot``).
    generate_report:
        In-memory generate report dict (from ``generate/run.py``).  If None,
        the function reads ``{out_dir}/stats/generate_report.json``.
    curate_stats:
        In-memory curate stats dict (from ``curate/run.py``).  If None,
        the function reads ``{out_dir}/stats.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building pilot report in: %s", out_dir)
    report = _build_report(out_dir, generate_report=generate_report, curate_stats=curate_stats)

    # Write JSON report
    json_path = out_dir / "pilot_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote: %s", json_path)

    # Write Markdown report
    md_path = out_dir / "pilot_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    logger.info("Wrote: %s", md_path)

    # Write examples JSONL
    examples_path = out_dir / "pilot_examples.jsonl"
    all_examples: list[dict[str, Any]] = []
    for category, items in report.get("examples", {}).items():
        for item in items:
            all_examples.append({"category": category, **item})
    with open(examples_path, "w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote: %s (%d examples)", examples_path, len(all_examples))
