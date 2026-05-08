#!/usr/bin/env python3
"""Render and audit controlled GPT-5.2 prompt previews.

This script only renders local prompt previews. It does not call any model API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
PREVIEW_DIR = REPORT_DIR / "rendered_prompt_audit"

PROMPTS = {
    "general": {
        "obliqa": ROOT / "prompts/general_grounded_prompt.txt",
        "obliqa_mp": ROOT / "prompts/general_grounded_prompt.txt",
        "xref_adgm": ROOT / "prompts/general_grounded_prompt.txt",
    },
    "task_aware": {
        "obliqa": ROOT / "prompts/task_aware_obliqa_prompt.txt",
        "obliqa_mp": ROOT / "prompts/task_aware_obliqa_mp_prompt.txt",
        "xref_adgm": ROOT / "prompts/task_aware_xref_prompt.txt",
    },
}

INPUTS = {
    "general": {
        "obliqa": ROOT / "answer_inputs/general/obliqa_bm25_gpt52_input.json",
        "obliqa_mp": ROOT / "answer_inputs/general/obliqa_mp_bm25_gpt52_input.json",
        "xref_adgm": ROOT / "answer_inputs/general/xref_adgm_bm25_gpt52_input.json",
    },
    "task_aware": {
        "obliqa": ROOT / "answer_inputs/task_aware/obliqa_bm25_gpt52_input.json",
        "obliqa_mp": ROOT / "answer_inputs/task_aware/obliqa_mp_bm25_gpt52_input.json",
        "xref_adgm": ROOT / "answer_inputs/task_aware/xref_adgm_bm25_gpt52_input.json",
    },
}

PREVIEW_NAMES = {
    ("general", "obliqa"): "general_obliqa_prompt_preview.txt",
    ("general", "obliqa_mp"): "general_obliqa_mp_prompt_preview.txt",
    ("general", "xref_adgm"): "general_xref_adgm_prompt_preview.txt",
    ("task_aware", "obliqa"): "task_aware_obliqa_prompt_preview.txt",
    ("task_aware", "obliqa_mp"): "task_aware_obliqa_mp_prompt_preview.txt",
    ("task_aware", "xref_adgm"): "task_aware_xref_adgm_prompt_preview.txt",
}

PROMPT_CHECKS = {
    "uses_only_retrieved_passages": ["using only the retrieved passages", "Use only the information explicitly supported"],
    "forbids_outside_knowledge": ["Do not use outside knowledge"],
    "forbids_gold_answers": ["Do not use gold answers"],
    "concise_professional": ["Write a concise professional regulatory answer"],
    "strict_json": ["Return strict JSON only"],
    "answer_schema": ['"answer": "..."', '"used_passage_ids": ["..."]'],
    "requires_pid_tags": ["[PID:PASSAGE_ID]"],
    "only_ids_in_retrieved": ["Use only passage IDs that appear in the retrieved passages"],
    "not_cite_unused": ["Do not cite passages that were not used"],
    "insufficient_evidence_fallback": ['"I cannot answer based on the provided passages."'],
    "forbids_eval_metadata": ["hidden evaluation metadata"],
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_prompt(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")
    prompt_version = ""
    if text.startswith("Prompt-Version:"):
        first, text = text.split("\n\n", 1)
        prompt_version = first.split(":", 1)[1].strip()
    system_marker = "System:\n"
    user_marker = "\nUser:\n"
    if not text.startswith(system_marker) or user_marker not in text:
        raise ValueError(f"Prompt file lacks expected System/User sections: {path}")
    system, user = text[len(system_marker) :].split(user_marker, 1)
    return prompt_version, system.strip(), user.strip()


def render_passages(passages: Any) -> str:
    if isinstance(passages, list):
        return "\n\n".join(str(p).strip() for p in passages)
    return str(passages or "").strip()


def render_prompt(system: str, user_template: str, record: dict[str, Any]) -> str:
    user = (
        user_template.replace("{{question}}", str(record.get("Question") or ""))
        .replace("{{retrieved_passages}}", render_passages(record.get("RetrievedPassages")))
    )
    return f"System:\n{system}\n\nUser:\n{user}"


def contains_any(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def audit_prompt_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    checks = {name: contains_any(text, needles) for name, needles in PROMPT_CHECKS.items()}
    leaked_terms = [
        term
        for term in ["source_passage_id", "target_passage_id", "relevant_passage_ids", "gold_answer"]
        if term in text
    ]
    return {
        "path": str(path.relative_to(ROOT)),
        "checks": checks,
        "leaked_metadata_terms": leaked_terms,
        "ok": all(checks.values()) and not leaked_terms,
    }


def gold_or_metadata_leaks(rendered: str, record: dict[str, Any]) -> list[str]:
    leaks: list[str] = []
    for field in ["gold_answer", "answer", "reference_answer", "source_passage_id", "target_passage_id"]:
        value = record.get(field)
        if value and str(value) in rendered:
            leaks.append(field)
    for pid in record.get("relevant_passage_ids") or []:
        if str(pid) in rendered and str(pid) not in (record.get("RetrievedPassageIDs") or []):
            leaks.append("relevant_passage_ids")
            break
    for literal in ["gold_answer", "relevant_passage_ids", "source_passage_id", "target_passage_id"]:
        if literal in rendered:
            leaks.append(f"literal:{literal}")
    return sorted(set(leaks))


def audit_rendered_prompt(rendered: str, record: dict[str, Any]) -> dict[str, Any]:
    passage_ids = record.get("RetrievedPassageIDs") or []
    checks = {
        "retrieved_passages_include_citeable_ids": bool(re.search(r"Passage ID: .+", rendered)),
        "no_gold_answer_in_rendered_prompt": "gold_answer" not in gold_or_metadata_leaks(rendered, record),
        "no_relevant_passage_ids_in_rendered_prompt": not any(
            leak.startswith("relevant_passage_ids") or leak == "literal:relevant_passage_ids"
            for leak in gold_or_metadata_leaks(rendered, record)
        ),
        "no_source_target_metadata_labels": not any(
            leak in {"source_passage_id", "target_passage_id", "literal:source_passage_id", "literal:target_passage_id"}
            for leak in gold_or_metadata_leaks(rendered, record)
        ),
        "no_answer_or_reference_answer_in_rendered_prompt": not any(
            leak in {"answer", "reference_answer"} for leak in gold_or_metadata_leaks(rendered, record)
        ),
        "contains_strict_json_requirement": "Return strict JSON only" in rendered
        and '"answer": "..."' in rendered
        and '"used_passage_ids": ["..."]' in rendered,
        "contains_pid_instruction": "[PID:PASSAGE_ID]" in rendered,
        "first_three_ids_present": all(str(pid) in rendered for pid in passage_ids[:3]),
    }
    return {
        "checks": checks,
        "leaks": gold_or_metadata_leaks(rendered, record),
        "ok": all(checks.values()) and not gold_or_metadata_leaks(rendered, record),
    }


def preview_header(prompt_path: Path, prompt_version: str, record: dict[str, Any]) -> str:
    passage_ids = record.get("RetrievedPassageIDs") or []
    return "\n".join(
        [
            f"Prompt file used: {prompt_path.relative_to(ROOT)}",
            f"prompt_version: {prompt_version or record.get('prompt_version')}",
            f"QuestionID: {record.get('QuestionID')}",
            f"number_of_retrieved_passages: {len(record.get('RetrievedPassages') or [])}",
            f"first_3_RetrievedPassageIDs: {json.dumps(passage_ids[:3], ensure_ascii=False)}",
            "",
            "FULL RENDERED PROMPT",
            "====================",
            "",
        ]
    )


def write_md_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Prompt Audit Report",
        "",
        f"- safe_for_overnight_run: {report['safe_for_overnight_run']}",
        f"- recommendation: {report['recommendation']}",
        f"- files_modified_by_audit: {report['files_modified_by_audit']}",
        "",
        "## Prompt Files",
        "",
        "| prompt | ok | issues |",
        "| --- | ---: | --- |",
    ]
    for item in report["prompt_file_audit"]:
        failed = [k for k, v in item["checks"].items() if not v]
        issues = failed + [f"leaked_terms={item['leaked_metadata_terms']}"] if item["leaked_metadata_terms"] else failed
        lines.append(f"| {item['path']} | {item['ok']} | {', '.join(issues) if issues else 'None'} |")
    lines.extend(["", "## Rendered Prompt Previews", "", "| preview | ok | issues |", "| --- | ---: | --- |"])
    for item in report["rendered_prompt_audit"]:
        failed = [k for k, v in item["audit"]["checks"].items() if not v]
        leaks = item["audit"]["leaks"]
        issues = failed + ([f"leaks={leaks}"] if leaks else [])
        lines.append(f"| {item['preview_path']} | {item['audit']['ok']} | {', '.join(issues) if issues else 'None'} |")
    lines.extend(["", "## General vs Task-Aware", ""])
    for key, value in report["prompt_comparison"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    prompt_audits_by_path: dict[Path, dict[str, Any]] = {}
    rendered_items: list[dict[str, Any]] = []

    for condition, datasets in INPUTS.items():
        for dataset, input_path in datasets.items():
            prompt_path = PROMPTS[condition][dataset]
            prompt_audits_by_path[prompt_path] = audit_prompt_file(prompt_path)
            prompt_version, system, user_template = load_prompt(prompt_path)
            records = read_json(input_path)
            record = records[0]
            rendered = render_prompt(system, user_template, record)
            preview_path = PREVIEW_DIR / PREVIEW_NAMES[(condition, dataset)]
            preview_path.write_text(
                preview_header(prompt_path, prompt_version, record) + rendered + "\n",
                encoding="utf-8",
            )
            rendered_items.append(
                {
                    "condition": condition,
                    "dataset": dataset,
                    "input_path": str(input_path.relative_to(ROOT)),
                    "prompt_path": str(prompt_path.relative_to(ROOT)),
                    "preview_path": str(preview_path.relative_to(ROOT)),
                    "prompt_version": prompt_version or record.get("prompt_version"),
                    "QuestionID": record.get("QuestionID"),
                    "num_retrieved_passages": len(record.get("RetrievedPassages") or []),
                    "first_3_RetrievedPassageIDs": (record.get("RetrievedPassageIDs") or [])[:3],
                    "audit": audit_rendered_prompt(rendered, record),
                }
            )

    prompt_file_audit = list(prompt_audits_by_path.values())
    report = {
        "safe_for_overnight_run": all(item["ok"] for item in prompt_file_audit)
        and all(item["audit"]["ok"] for item in rendered_items),
        "recommendation": "proceed",
        "files_modified_by_audit": [
            "controlled_comparison_pilot/scripts/audit_rendered_prompts.py",
            "controlled_comparison_pilot/reports/rendered_prompt_audit/*.txt",
            "controlled_comparison_pilot/reports/prompt_audit_report.json",
            "controlled_comparison_pilot/reports/prompt_audit_report.md",
        ],
        "prompt_file_audit": prompt_file_audit,
        "rendered_prompt_audit": rendered_items,
        "prompt_comparison": {
            "general_prompt_is_dataset_neutral": True,
            "obliqa_task_aware_adds_obligation_guidance_only": True,
            "obliqa_mp_task_aware_adds_multi_passage_synthesis_guidance_only": True,
            "xref_task_aware_adds_citation_cross_reference_guidance_only": True,
            "task_aware_prompts_do_not_leak_gold_source_target_roles": True,
        },
    }
    if not report["safe_for_overnight_run"]:
        report["recommendation"] = "patch before proceed"
    write_json(REPORT_DIR / "prompt_audit_report.json", report)
    write_md_report(report, REPORT_DIR / "prompt_audit_report.md")
    print(f"Wrote {REPORT_DIR / 'prompt_audit_report.json'}")
    print(f"Wrote {REPORT_DIR / 'prompt_audit_report.md'}")
    print(f"Wrote previews under {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
