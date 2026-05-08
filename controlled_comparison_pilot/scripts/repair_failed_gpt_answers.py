#!/usr/bin/env python3
"""Repair failed GPT-5.2 answer records in the sample_300 batch.

Retries only records whose existing answer output is failed. Defaults to dry-run
and never creates an Azure OpenAI client unless --run is explicitly provided.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / "batch_runs/sample_300"
PROVIDER = "azure"
MAX_OUTPUT_TOKENS = 700
REPAIR_INSTRUCTION = (
    "Keep the answer concise. Return only a complete valid JSON object. "
    "Do not exceed 180 words."
)
ENV_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_GPT52",
]
DATASETS = {
    "obliqa": "obliqa_bm25_gpt52",
    "obliqa_mp": "obliqa_mp_bm25_gpt52",
    "xref_adgm": "xref_adgm_bm25_gpt52",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--dry_run",
        action="store_true",
        default=True,
        help="Inspect failures and write a report without calling the API. This is the default.",
    )
    run_group.add_argument(
        "--run",
        dest="dry_run",
        action="store_false",
        help="Explicitly permit Azure OpenAI retry calls.",
    )
    parser.add_argument("--batch_root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--max_retries_per_record", type=int, default=2)
    parser.add_argument("--log_progress_every", type=int, default=5)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[:-7]
    return endpoint


def env_status() -> dict[str, str]:
    return {name: "SET" if os.getenv(name, "").strip() else "MISSING" for name in ENV_VARS}


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def selected_files(batch_root: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for condition in ("general", "task_aware"):
        for dataset_key, stem in DATASETS.items():
            plan.append(
                {
                    "condition": condition,
                    "dataset_key": dataset_key,
                    "stem": stem,
                    "input_path": batch_root / "answer_inputs" / condition / f"{stem}_input.json",
                    "answer_path": batch_root / "answers" / condition / f"{stem}_answers.json",
                    "prompt_path": PROMPTS[condition][dataset_key],
                }
            )
    return plan


def load_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("Prompt-Version:"):
        parts = text.split("\n\n", 1)
        text = parts[1] if len(parts) == 2 else ""
    system_marker = "System:\n"
    user_marker = "\nUser:\n"
    if not text.startswith(system_marker) or user_marker not in text:
        raise ValueError(f"Prompt file does not contain expected System/User sections: {path}")
    system_part, user_part = text[len(system_marker) :].split(user_marker, 1)
    return system_part.strip(), user_part.strip()


def render_passages(passages: Any) -> str:
    if isinstance(passages, list):
        return "\n\n".join(str(p).strip() for p in passages)
    return str(passages or "").strip()


def render_user_prompt(template: str, record: dict[str, Any]) -> str:
    user = (
        template.replace("{{question}}", str(record.get("Question") or ""))
        .replace("{{retrieved_passages}}", render_passages(record.get("RetrievedPassages")))
    )
    return f"{user}\n\nAdditional repair instruction:\n{REPAIR_INSTRUCTION}"


def balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def json_candidates(raw: str) -> list[str]:
    stripped = raw.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1]).strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            candidates.append(inner)
    balanced = balanced_json_object(stripped)
    if balanced:
        candidates.append(balanced)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def parse_answer(raw: str) -> tuple[bool, str, list[str], str]:
    last_error = "json_parse_error: empty model output"
    for candidate in json_candidates(raw):
        try:
            obj = json.loads(candidate)
        except Exception as exc:
            last_error = f"json_parse_error: {exc}"
            continue
        if not isinstance(obj, dict):
            return False, "", [], "json_parse_error: output is not a JSON object"
        answer = obj.get("answer")
        used_passage_ids = obj.get("used_passage_ids", [])
        if not isinstance(answer, str):
            return False, "", [], "json_parse_error: missing string field 'answer'"
        if not isinstance(used_passage_ids, list):
            return False, "", [], "json_parse_error: used_passage_ids is not a list"
        used_passage_ids = [str(pid) for pid in used_passage_ids]
        return True, answer.strip(), used_passage_ids, ""
    return False, "", [], last_error


def usage_dict(resp: Any) -> dict[str, int | None]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def create_client(endpoint: str, api_key: str, api_version: str) -> Any:
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=normalize_endpoint(endpoint),
        api_key=api_key,
        api_version=api_version,
    )


def call_model(client: Any, *, deployment: str, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int | None]]:
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    return raw, usage_dict(resp)


def needs_retry(record: dict[str, Any]) -> bool:
    return (
        record.get("json_parse_ok") is False
        or bool(str(record.get("error") or "").strip())
        or not str(record.get("generated_answer_with_grounding") or "").strip()
    )


def repaired_record(
    old: dict[str, Any],
    input_record: dict[str, Any],
    *,
    raw: str,
    answer: str,
    used_passage_ids: list[str],
    usage: dict[str, int | None],
    deployment: str,
    api_version: str,
    run_date: str,
) -> dict[str, Any]:
    updated = dict(old)
    updated.update(
        {
            "QuestionID": input_record.get("QuestionID"),
            "Question": input_record.get("Question"),
            "RetrievedPassages": input_record.get("RetrievedPassages"),
            "RetrievedPassageIDs": input_record.get("RetrievedPassageIDs"),
            "generated_answer_with_grounding": answer,
            "used_passage_ids": used_passage_ids,
            "raw_model_output": raw,
            "dataset": input_record.get("dataset"),
            "split": input_record.get("split"),
            "sample_seed": input_record.get("sample_seed"),
            "sample_size": input_record.get("sample_size"),
            "retriever": input_record.get("retriever"),
            "retrieval_setting": input_record.get("retrieval_setting"),
            "retrieval_corpus": input_record.get("retrieval_corpus"),
            "k": input_record.get("k"),
            "prompt_condition": input_record.get("prompt_condition"),
            "prompt_version": input_record.get("prompt_version"),
            "actual_model_or_deployment": deployment,
            "provider": PROVIDER,
            "api_version": api_version,
            "run_date": run_date,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "json_parse_ok": True,
            "error": "",
            "repair_run_date": run_date,
            "repaired": True,
        }
    )
    return updated


def backup_file(path: Path, timestamp: str) -> Path:
    backup_path = path.with_suffix(path.suffix + f".bak_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def report_paths(batch_root: Path) -> tuple[Path, Path]:
    return (
        batch_root / "reports/gpt52_repair_report.json",
        batch_root / "reports/gpt52_repair_report.md",
    )


def write_md_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# GPT-5.2 Repair Report",
        "",
        f"- dry_run: {report['dry_run']}",
        f"- start_time: {report['start_time']}",
        f"- end_time: {report['end_time']}",
        f"- records_needing_retry: {report['records_needing_retry']}",
        f"- retry_calls_attempted: {report['retry_calls_attempted']}",
        f"- records_repaired: {report['records_repaired']}",
        f"- records_still_failed: {report['records_still_failed']}",
        "",
        "| answer_file | records | needing_retry | repaired | still_failed | backup |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["files"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["answer_path"],
                    str(item["records"]),
                    str(item["needing_retry"]),
                    str(item["repaired"]),
                    str(item["still_failed"]),
                    item.get("backup_path") or "",
                ]
            )
            + " |"
        )
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    write_text(path, "\n".join(lines) + "\n")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    batch_root = args.batch_root
    start = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    timestamp = start.strftime("%Y%m%d_%H%M%S")
    run_date = start.isoformat()
    json_report_path, md_report_path = report_paths(batch_root)
    warnings: list[str] = []

    print("Environment variable status:")
    for name, status in env_status().items():
        print(f"  {name}: {status}")

    plan = selected_files(batch_root)
    report_files: list[dict[str, Any]] = []
    total_needing_retry = 0
    total_attempted = 0
    total_repaired = 0
    total_still_failed = 0
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    token_seen = {"prompt_tokens": False, "completion_tokens": False, "total_tokens": False}

    client = None
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "").strip() or "MISSING"
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or "MISSING"
    if not args.dry_run:
        endpoint = require_env("AZURE_OPENAI_ENDPOINT")
        api_key = require_env("AZURE_OPENAI_API_KEY")
        api_version = require_env("AZURE_OPENAI_API_VERSION")
        deployment = require_env("AZURE_OPENAI_DEPLOYMENT_GPT52")
        client = create_client(endpoint, api_key, api_version)

    for item in plan:
        answer_path = item["answer_path"]
        input_path = item["input_path"]
        rows = read_json(answer_path)
        inputs = {str(row.get("QuestionID")): row for row in read_json(input_path)}
        failed_indices = [
            idx
            for idx, row in enumerate(rows)
            if isinstance(row, dict) and needs_retry(row)
        ]
        total_needing_retry += len(failed_indices)
        file_repaired = 0
        file_still_failed = len(failed_indices)
        backup_path = ""

        print(f"{rel(answer_path)}: {len(failed_indices)} records need retry", flush=True)
        if failed_indices and not args.dry_run:
            backup_path = rel(backup_file(answer_path, timestamp))
            system_prompt, user_template = load_prompt(item["prompt_path"])

            for failed_no, idx in enumerate(failed_indices, start=1):
                old = rows[idx]
                qid = str(old.get("QuestionID") or "")
                input_record = inputs.get(qid)
                if input_record is None:
                    warnings.append(f"{rel(answer_path)}: missing input record for QuestionID={qid}")
                    continue

                repaired = False
                last_raw = ""
                last_error = ""
                last_usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                for _attempt in range(args.max_retries_per_record):
                    total_attempted += 1
                    try:
                        assert client is not None
                        raw, usage = call_model(
                            client,
                            deployment=deployment,
                            system_prompt=system_prompt,
                            user_prompt=render_user_prompt(user_template, input_record),
                        )
                        last_raw = raw
                        last_usage = usage
                        for key in token_totals:
                            val = usage.get(key)
                            if isinstance(val, int):
                                token_totals[key] += val
                                token_seen[key] = True
                        ok, answer, used_passage_ids, parse_error = parse_answer(raw)
                        if ok and answer:
                            rows[idx] = repaired_record(
                                old,
                                input_record,
                                raw=raw,
                                answer=answer,
                                used_passage_ids=used_passage_ids,
                                usage=usage,
                                deployment=deployment,
                                api_version=api_version,
                                run_date=run_date,
                            )
                            repaired = True
                            file_repaired += 1
                            total_repaired += 1
                            break
                        last_error = parse_error or "empty answer"
                    except Exception as exc:
                        last_error = str(exc)

                if not repaired:
                    updated = dict(old)
                    updated.update(
                        {
                            "raw_model_output": last_raw or old.get("raw_model_output", ""),
                            "prompt_tokens": last_usage.get("prompt_tokens"),
                            "completion_tokens": last_usage.get("completion_tokens"),
                            "total_tokens": last_usage.get("total_tokens"),
                            "json_parse_ok": False,
                            "error": last_error or old.get("error") or "repair_failed",
                            "repair_run_date": run_date,
                            "repair_attempted": True,
                        }
                    )
                    rows[idx] = updated

                if args.log_progress_every > 0 and failed_no % args.log_progress_every == 0:
                    print(
                        f"  progress {failed_no}/{len(failed_indices)} in {rel(answer_path)}; "
                        f"total_attempted={total_attempted}, total_repaired={total_repaired}",
                        flush=True,
                    )

            write_json(answer_path, rows)
            file_still_failed = sum(1 for row in rows if isinstance(row, dict) and needs_retry(row))

        report_files.append(
            {
                "condition": item["condition"],
                "dataset_key": item["dataset_key"],
                "input_path": rel(input_path),
                "answer_path": rel(answer_path),
                "records": len(rows),
                "needing_retry": len(failed_indices),
                "repaired": file_repaired,
                "still_failed": file_still_failed if not args.dry_run else len(failed_indices),
                "backup_path": backup_path,
                "failed_question_ids": [rows[idx].get("QuestionID") for idx in failed_indices],
            }
        )

    total_still_failed = sum(item["still_failed"] for item in report_files)
    end = datetime.now(timezone.utc)
    report = {
        "dry_run": args.dry_run,
        "batch_root": rel(batch_root),
        "start_time": run_date,
        "end_time": end.isoformat(),
        "runtime_seconds": time.perf_counter() - start_perf,
        "max_retries_per_record": args.max_retries_per_record,
        "records_needing_retry": total_needing_retry,
        "retry_calls_attempted": total_attempted,
        "records_repaired": total_repaired,
        "records_still_failed": total_still_failed,
        "token_totals": {key: token_totals[key] if token_seen[key] else None for key in token_totals},
        "actual_model_or_deployment": deployment,
        "provider": PROVIDER,
        "api_version": api_version,
        "files": report_files,
        "warnings": warnings,
    }
    write_json(json_report_path, report)
    write_md_report(md_report_path, report)
    print(f"Wrote {rel(json_report_path)}")
    print(f"Wrote {rel(md_report_path)}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
