#!/usr/bin/env python3
"""Run the GPT-5.2 controlled-comparison answer pilot.

Defaults to dry-run validation. Real Azure OpenAI calls require --run.
This script does not run RePASs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_TOKENS = 700
REPORT_PATH = ROOT / "reports/gpt52_pilot_run_report.json"
PROVIDER = "azure"
ENV_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_GPT52",
]
REQUIRED_INPUT_FIELDS = [
    "Question",
    "RetrievedPassages",
    "RetrievedPassageIDs",
    "prompt_condition",
    "prompt_version",
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
    parser = argparse.ArgumentParser(
        description=(
            "Generate GPT-5.2 pilot answers for the controlled comparison. "
            "The default mode is dry-run validation; use --run for API calls."
        )
    )
    parser.add_argument(
        "--prompt_condition",
        choices=["general", "task_aware", "both"],
        default="both",
        help="Prompt condition to process.",
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--dry_run",
        action="store_true",
        default=True,
        help="Validate inputs and print the planned run without calling the API. This is the default.",
    )
    run_group.add_argument(
        "--run",
        dest="dry_run",
        action="store_false",
        help="Explicitly permit Azure OpenAI API calls.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing answer output files. Without this, existing answer files are skipped.",
    )
    parser.add_argument("--input_root", type=Path, default=ROOT / "answer_inputs")
    parser.add_argument("--output_root", type=Path, default=ROOT / "answers")
    parser.add_argument("--report_path", type=Path, default=REPORT_PATH)
    parser.add_argument("--log_progress_every", type=int, default=10)
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing output files by skipping records already present by QuestionID.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[:-7]
    return endpoint


def env_status() -> dict[str, str]:
    return {name: "SET" if os.getenv(name, "").strip() else "MISSING" for name in ENV_VARS}


def print_env_status() -> None:
    print("Environment variable status:")
    for name, status in env_status().items():
        print(f"  {name}: {status}")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def selected_conditions(prompt_condition: str) -> list[str]:
    if prompt_condition == "both":
        return ["general", "task_aware"]
    return [prompt_condition]


def input_path(input_root: Path, condition: str, dataset_name: str) -> Path:
    return input_root / condition / f"{DATASETS[dataset_name]}_input.json"


def output_path(output_root: Path, condition: str, dataset_name: str) -> Path:
    return output_root / condition / f"{DATASETS[dataset_name]}_answers.json"


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
    return (
        template.replace("{{question}}", str(record.get("Question") or ""))
        .replace("{{retrieved_passages}}", render_passages(record.get("RetrievedPassages")))
    )


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
            candidates.append("\n".join(lines[1:-1]).strip())
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
        used_passage_ids = obj.get("used_passage_ids")
        if not isinstance(answer, str):
            return False, "", [], "json_parse_error: missing string field 'answer'"
        if not isinstance(used_passage_ids, list) or not all(
            isinstance(pid, str) for pid in used_passage_ids
        ):
            return False, "", [], "json_parse_error: missing string list field 'used_passage_ids'"
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


def call_model(
    client: Any,
    *,
    deployment: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, int | None]]:
    kwargs: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    return raw, usage_dict(resp)


def output_record(
    record: dict[str, Any],
    *,
    generated_answer: str,
    used_passage_ids: list[str],
    raw_output: str,
    deployment: str,
    api_version: str,
    run_date: str,
    usage: dict[str, int | None] | None,
    json_parse_ok: bool,
    error: str,
) -> dict[str, Any]:
    usage = usage or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "QuestionID": record.get("QuestionID"),
        "Question": record.get("Question"),
        "RetrievedPassages": record.get("RetrievedPassages"),
        "RetrievedPassageIDs": record.get("RetrievedPassageIDs"),
        "generated_answer_with_grounding": generated_answer,
        "used_passage_ids": used_passage_ids,
        "raw_model_output": raw_output,
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "sample_seed": record.get("sample_seed"),
        "sample_size": record.get("sample_size"),
        "retriever": record.get("retriever"),
        "retrieval_setting": record.get("retrieval_setting"),
        "retrieval_corpus": record.get("retrieval_corpus"),
        "k": record.get("k"),
        "prompt_condition": record.get("prompt_condition"),
        "prompt_version": record.get("prompt_version"),
        "actual_model_or_deployment": deployment,
        "provider": PROVIDER,
        "api_version": api_version,
        "run_date": run_date,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "json_parse_ok": json_parse_ok,
        "error": error,
    }


def validate_record(record: Any, path: Path, idx: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{path}: record {idx} is not a JSON object"]
    for field in REQUIRED_INPUT_FIELDS:
        if field not in record:
            errors.append(f"{path}: record {idx} missing {field}")
    return errors


def validate_inputs(plan: list[dict[str, Any]]) -> tuple[int, list[str]]:
    errors: list[str] = []
    total_records = 0
    for item in plan:
        prompt_path = item["prompt_path"]
        in_path = item["input_path"]
        if not prompt_path.exists():
            errors.append(f"Missing prompt file: {prompt_path}")
        if not in_path.exists():
            errors.append(f"Missing input file: {in_path}")
            continue
        try:
            records = read_json(in_path)
        except Exception as exc:
            errors.append(f"Could not read input file {in_path}: {exc}")
            continue
        if not isinstance(records, list):
            errors.append(f"{in_path}: expected a JSON list")
            continue
        total_records += len(records)
        for idx, record in enumerate(records, start=1):
            errors.extend(validate_record(record, in_path, idx))
    return total_records, errors


def build_plan(prompt_condition: str, input_root: Path, output_root: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for condition in selected_conditions(prompt_condition):
        for dataset_name in DATASETS:
            plan.append(
                {
                    "condition": condition,
                    "dataset_name": dataset_name,
                    "input_path": input_path(input_root, condition, dataset_name),
                    "prompt_path": PROMPTS[condition][dataset_name],
                    "output_path": output_path(output_root, condition, dataset_name),
                }
            )
    return plan


def report_base(
    *,
    dry_run: bool,
    requested_prompt_condition: str,
    start: datetime,
    end: datetime,
    runtime_seconds: float,
    num_calls_planned: int,
    output_paths: list[str],
    deployment: str,
    api_version: str,
    num_calls_attempted: int = 0,
    num_successful: int = 0,
    num_failed: int = 0,
    token_totals: dict[str, int | None] | None = None,
    average_time_per_call: float | None = None,
    json_parse_failures: int = 0,
    empty_generated_answers: int = 0,
    insufficient_evidence_count: int = 0,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "prompt_condition_requested": requested_prompt_condition,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "runtime_seconds": runtime_seconds,
        "num_calls_planned": num_calls_planned,
        "num_calls_attempted": num_calls_attempted,
        "num_successful": num_successful,
        "num_failed": num_failed,
        "token_totals": token_totals
        or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "average_time_per_call": average_time_per_call,
        "JSON_parse_failures": json_parse_failures,
        "empty_generated_answers": empty_generated_answers,
        "insufficient_evidence_answer_count": insufficient_evidence_count,
        "output_file_paths": output_paths,
        "actual_model_or_deployment": deployment,
        "model_deployment": deployment,
        "provider": PROVIDER,
        "api_version": api_version,
    }


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def existing_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        rows = read_json(path)
    except Exception:
        return set()
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("QuestionID"))
        for row in rows
        if isinstance(row, dict) and row.get("QuestionID") and row.get("json_parse_ok") is not None
    }


def read_existing_outputs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_json(path)
    return rows if isinstance(rows, list) else []


def print_plan(plan: list[dict[str, Any]], *, overwrite: bool, resume: bool) -> None:
    print("Planned files:")
    for item in plan:
        exists = "exists" if item["output_path"].exists() else "new"
        action = "overwrite" if overwrite and item["output_path"].exists() else "write"
        if item["output_path"].exists() and resume and not overwrite:
            action = "resume"
        elif item["output_path"].exists() and not overwrite:
            action = "skip existing output"
        print(f"  [{item['condition']} / {item['dataset_name']}]")
        print(f"    input:  {rel(item['input_path'])}")
        print(f"    prompt: {rel(item['prompt_path'])}")
        print(f"    output: {rel(item['output_path'])} ({exists}; {action})")


def create_client(endpoint: str, api_key: str, api_version: str) -> Any:
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=normalize_endpoint(endpoint),
        api_key=api_key,
        api_version=api_version,
    )


def run_dry_run(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    start = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    print(f"Selected prompt condition: {args.prompt_condition}", flush=True)
    print_env_status()
    print_plan(plan, overwrite=args.overwrite, resume=args.resume)
    total_records, errors = validate_inputs(plan)
    skipped_records = 0
    output_paths: list[str] = []
    for item in plan:
        output_paths.append(rel(item["output_path"]))
        records = read_json(item["input_path"]) if item["input_path"].exists() else []
        if item["output_path"].exists() and args.resume and not args.overwrite:
            skipped_records += len(existing_completed_ids(item["output_path"]))
        elif item["output_path"].exists() and not args.overwrite:
            skipped_records += len(records) if isinstance(records, list) else 0
    planned_calls = total_records - skipped_records
    if args.max_calls is not None:
        planned_calls = min(planned_calls, args.max_calls)
    print(f"Total input records selected: {total_records}")
    print(f"Planned API calls in this mode: {planned_calls}")
    if skipped_records:
        print(f"Skipped records due to existing output files: {skipped_records}")
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Validation OK.")
    end = datetime.now(timezone.utc)
    report = report_base(
        dry_run=True,
        requested_prompt_condition=args.prompt_condition,
        start=start,
        end=end,
        runtime_seconds=time.perf_counter() - start_perf,
        num_calls_planned=planned_calls,
        output_paths=output_paths,
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "").strip() or "MISSING",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or "MISSING",
    )
    write_json(args.report_path, report)
    if errors:
        raise SystemExit(1)


def run_real(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    start = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    print(f"Selected prompt condition: {args.prompt_condition}", flush=True)
    print_env_status()
    total_records, errors = validate_inputs(plan)
    if errors:
        for error in errors:
            print(f"Validation error: {error}")
        raise SystemExit(1)

    endpoint = require_env("AZURE_OPENAI_ENDPOINT")
    api_key = require_env("AZURE_OPENAI_API_KEY")
    api_version = require_env("AZURE_OPENAI_API_VERSION")
    deployment = require_env("AZURE_OPENAI_DEPLOYMENT_GPT52")
    client = create_client(endpoint, api_key, api_version)

    run_date = start.isoformat()
    num_calls_planned = 0
    for item in plan:
        if item["output_path"].exists() and not args.overwrite and not args.resume:
            continue
        records = read_json(item["input_path"])
        completed = existing_completed_ids(item["output_path"]) if args.resume and not args.overwrite else set()
        num_calls_planned += sum(1 for record in records if str(record.get("QuestionID")) not in completed)
    if args.max_calls is not None:
        num_calls_planned = min(num_calls_planned, args.max_calls)

    num_calls_attempted = 0
    num_successful = 0
    num_failed = 0
    json_parse_failures = 0
    empty_generated_answers = 0
    insufficient_evidence_count = 0
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    token_seen = {"prompt_tokens": False, "completion_tokens": False, "total_tokens": False}
    call_times: list[float] = []
    output_paths: list[str] = []
    global_completed = 0

    for item in plan:
        out_path = item["output_path"]
        output_paths.append(rel(out_path))
        if out_path.exists() and not args.overwrite and not args.resume:
            print(f"Skipping existing output: {rel(out_path)}", flush=True)
            continue

        system_prompt, user_template = load_prompt(item["prompt_path"])
        inputs = read_json(item["input_path"])
        existing_outputs = [] if args.overwrite else read_existing_outputs(out_path)
        completed_ids = {
            str(row.get("QuestionID"))
            for row in existing_outputs
            if isinstance(row, dict) and row.get("QuestionID") and row.get("json_parse_ok") is not None
        }
        outputs: list[dict[str, Any]] = list(existing_outputs)
        print(
            f"Starting {item['condition']} / {item['dataset_name']}: "
            f"{rel(item['input_path'])} -> {rel(out_path)}",
            flush=True,
        )

        for record in inputs:
            if args.resume and str(record.get("QuestionID")) in completed_ids:
                continue
            if args.max_calls is not None and num_calls_attempted >= args.max_calls:
                print(f"Reached --max_calls={args.max_calls}; stopping.", flush=True)
                break
            user_prompt = render_user_prompt(user_template, record)
            call_start = time.perf_counter()
            num_calls_attempted += 1
            raw = ""
            usage: dict[str, int | None] | None = None
            generated_answer = ""
            used_passage_ids: list[str] = []
            parse_ok = False
            error = ""
            try:
                raw, usage = call_model(
                    client,
                    deployment=deployment,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                parse_ok, generated_answer, used_passage_ids, error = parse_answer(raw)
                for key in token_totals:
                    val = usage.get(key)
                    if isinstance(val, int):
                        token_totals[key] += val
                        token_seen[key] = True
                if parse_ok:
                    num_successful += 1
                    if not generated_answer.strip():
                        empty_generated_answers += 1
                    if generated_answer.strip() == "I cannot answer based on the provided passages.":
                        insufficient_evidence_count += 1
                else:
                    num_failed += 1
                    json_parse_failures += 1
            except Exception as exc:
                num_failed += 1
                error = str(exc)
            finally:
                call_times.append(time.perf_counter() - call_start)

            outputs.append(
                output_record(
                    record,
                    generated_answer=generated_answer,
                    used_passage_ids=used_passage_ids,
                    raw_output=raw,
                    deployment=deployment,
                    api_version=api_version,
                    run_date=run_date,
                    usage=usage,
                    json_parse_ok=parse_ok,
                    error=error,
                )
            )
            global_completed += 1
            if args.log_progress_every > 0 and global_completed % args.log_progress_every == 0:
                elapsed = time.perf_counter() - start_perf
                print(
                    f"Progress: completed {global_completed}/{num_calls_planned}, "
                    f"successes={num_successful}, failures={num_failed}, elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )

        write_json(out_path, outputs)
        print(f"Completed {item['condition']} / {item['dataset_name']}: wrote {len(outputs)} records", flush=True)
        if args.max_calls is not None and num_calls_attempted >= args.max_calls:
            break

    end = datetime.now(timezone.utc)
    report = report_base(
        dry_run=False,
        requested_prompt_condition=args.prompt_condition,
        start=start,
        end=end,
        runtime_seconds=time.perf_counter() - start_perf,
        num_calls_planned=num_calls_planned,
        output_paths=output_paths,
        deployment=deployment,
        api_version=api_version,
        num_calls_attempted=num_calls_attempted,
        num_successful=num_successful,
        num_failed=num_failed,
        token_totals={key: token_totals[key] if token_seen[key] else None for key in token_totals},
        average_time_per_call=(sum(call_times) / len(call_times)) if call_times else None,
        json_parse_failures=json_parse_failures,
        empty_generated_answers=empty_generated_answers,
        insufficient_evidence_count=insufficient_evidence_count,
    )
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    plan = build_plan(args.prompt_condition, args.input_root, args.output_root)
    if args.dry_run:
        run_dry_run(args, plan)
    else:
        run_real(args, plan)


if __name__ == "__main__":
    main()
