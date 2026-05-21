#!/usr/bin/env python3
"""Run one LLM-as-a-judge pass over a human annotation subset CSV.

The LLM behaves like one annotator: the output CSV keeps the same schema as the
input CSV and fills the existing annotation columns. The input file is not
modified. By default, the output filename includes the model/deployment name.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(".env"), override=False)
except Exception:
    pass


DEFAULT_ADGM_INPUT = Path(
    "ObliQA-XRef_Out_Datasets/final_large_merged/human_verification/"
    "human_verification_subset_ADGM_100.csv"
)
ANNOTATION_COLS = [
    "question_validity_clarity",
    "question_fluency_naturalness",
    "source_target_evidence_dependency",
    "comment",
]


class ProviderQuotaExceeded(RuntimeError):
    """Raised when a provider quota remains exhausted after retries."""


SYSTEM_PROMPT = """You are a careful regulatory QA audit judge.

Use only the provided question, source passage, and target passage.
Return strict JSON only. Do not add extra text.
"""


def build_user_prompt(row: dict[str, str]) -> str:
    return f"""Audit this citation-dependent QA item using the exact label sets below.

Q1. Is the question understandable?
Allowed labels:
- yes
- partial_unclear
- no

Q2. Is the question natural, or does it rely too heavily on copied passage wording?
Allowed labels:
- mostly_natural_rephrased
- some_copied_acceptable
- too_much_copied_unnatural
- unclear

Q3. Which passage(s) are needed to answer the question?
Allowed labels:
- both_needed
- source_only_sufficient
- target_only_sufficient
- neither_mismatch_unclear

Question:
{row.get("question", "")}

Source passage:
{row.get("source_text", "")}

Target passage:
{row.get("target_text", "")}

Return strict JSON only with this shape:
{{
  "question_validity_clarity": "...",
  "question_fluency_naturalness": "...",
  "source_target_evidence_dependency": "...",
  "confidence": 0.0,
  "rationale": "one short sentence"
}}
"""


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_lenient(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = JSON_RE.search(text)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("Model response did not contain a JSON object")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def parse_retry_delay_s(text: str, default: float = 30.0) -> float:
    match = re.search(r"retry in ([0-9.]+)s", text or "", flags=re.I)
    if not match:
        return default
    try:
        return max(default, float(match.group(1)) + 2.0)
    except ValueError:
        return default


def get_client():
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: install openai to run the LLM judge.") from exc

    endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT", "") or "").strip().rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[:-7]
    api_key = (os.environ.get("AZURE_OPENAI_API_KEY", "") or "").strip()
    api_version = (os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview") or "").strip()
    if not endpoint or not api_key:
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY before running.")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=60,
        max_retries=0,
    )


def get_openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: install openai to run the LLM judge.") from exc

    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    api_key = api_key.strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key for provider=openai.")
    return OpenAI(api_key=api_key)


def normalize_anthropic_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/v1/messages"):
        return endpoint
    if endpoint.endswith("/anthropic"):
        return f"{endpoint}/v1/messages"
    return f"{endpoint}/anthropic/v1/messages"


def get_foundry_anthropic_config(args: argparse.Namespace) -> tuple[str, str, str]:
    endpoint = (
        args.endpoint
        or os.getenv("AZURE_AI_FOUNDRY_ANTHROPIC_ENDPOINT")
        or os.getenv("ANTHROPIC_AZURE_ENDPOINT")
        or "https://claude-fsra-resource.services.ai.azure.com/anthropic/"
    )
    api_key = (
        args.api_key
        or os.getenv("AZURE_AI_FOUNDRY_API_KEY")
        or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("ANTHROPIC_AZURE_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or ""
    ).strip()
    deployment = (
        args.deployment
        or os.getenv("AZURE_AI_FOUNDRY_ANTHROPIC_DEPLOYMENT")
        or os.getenv("ANTHROPIC_AZURE_DEPLOYMENT")
        or "claude-opus-4-5-MBZUAI"
    ).strip()
    if not api_key:
        raise SystemExit(
            "Set --api-key or AZURE_AI_FOUNDRY_API_KEY/ANTHROPIC_FOUNDRY_API_KEY/AZURE_API_KEY for Claude."
        )
    return normalize_anthropic_endpoint(endpoint), api_key, deployment


def preflight_azure_config(deployment: str) -> None:
    endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT", "") or "").strip().rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[:-7]
    api_key = (os.environ.get("AZURE_OPENAI_API_KEY", "") or "").strip()
    api_version = (os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview") or "").strip()

    if not endpoint or not api_key:
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY before running.")
    if not deployment:
        raise SystemExit("Provide --deployment or set AZURE_OPENAI_DEPLOYMENT_GPT52.")

    host = urlparse(endpoint).hostname
    print(f"Azure endpoint: {endpoint}")
    print(f"Azure deployment: {deployment}")
    print(f"Azure API version: {api_version}")
    print(f"Azure API key set: {'yes' if api_key else 'no'}")

    if not host:
        raise SystemExit(f"Invalid AZURE_OPENAI_ENDPOINT: {endpoint}")
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror as exc:
        print(
            "Warning: local DNS could not resolve AZURE_OPENAI_ENDPOINT hostname. "
            f"Host={host!r}; error={exc}. Continuing anyway because some environments "
            "resolve Azure endpoints through proxy/VPN settings used by the HTTP client."
        )


def call_judge(client: Any, deployment: str, row: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
                temperature=0,
                max_completion_tokens=500,
            )
            content = response.choices[0].message.content or ""
            return parse_json_lenient(content)
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_error}")


def call_openai_judge(client: Any, model: str, row: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
                temperature=0,
                max_completion_tokens=500,
            )
            content = response.choices[0].message.content or ""
            return parse_json_lenient(content)
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"OpenAI judge call failed after {max_retries} attempts: {last_error}")


def resolve_openai_model(args: argparse.Namespace) -> str:
    return (args.model or args.deployment or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()


def resolve_gemini_model(args: argparse.Namespace) -> str:
    return (
        args.model
        or args.deployment
        or os.getenv("GEMINI_MODEL")
        or "gemini-3-flash-preview"
    ).strip()


def get_gemini_api_key(args: argparse.Namespace) -> str:
    api_key = (
        args.api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY/GOOGLE_API_KEY or pass --api-key for provider=gemini.")
    return api_key


def gemini_generation_config(model: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": 0,
        "maxOutputTokens": 4096,
    }
    # Gemini 3 preview can spend the response budget on hidden thinking unless
    # we lower the thinking level. Gemini 2.5 Pro rejects this parameter.
    if model.startswith("gemini-3"):
        config["thinkingConfig"] = {"thinkingLevel": "minimal"}
    return config


def call_gemini_judge(
    *,
    api_key: str,
    model: str,
    row: dict[str, str],
    endpoint: str = "",
    max_retries: int = 30,
) -> dict[str, Any]:
    last_error: Exception | None = None
    base = (endpoint or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base}/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_user_prompt(row)
                        + "\n\nOutput only the JSON object. Do not use markdown, prose, or code fences."
                    }
                ],
            }
        ],
        "generationConfig": gemini_generation_config(model),
    }
    for attempt in range(max_retries):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=90)
            if response.status_code >= 400:
                if response.status_code == 429:
                    delay = parse_retry_delay_s(response.text)
                    last_error = RuntimeError(f"HTTP 429 Too Many Requests: {response.text[:800]}")
                    print(f"Gemini rate limit hit; sleeping {delay:.1f}s before retry")
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"HTTP {response.status_code} {response.reason_phrase}: {response.text[:800]}"
                )
            data = response.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))
            if not text:
                raise RuntimeError(f"Empty Gemini text response: {json.dumps(data)[:800]}")
            try:
                return parse_json_lenient(text)
            except Exception as exc:
                raise RuntimeError(f"Could not parse Gemini text as JSON. Text preview: {text[:800]!r}") from exc
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, 2**attempt))
    if last_error and "HTTP 429 Too Many Requests" in str(last_error):
        raise ProviderQuotaExceeded(
            f"Gemini quota remained exhausted after {max_retries} attempts: {last_error}"
        ) from last_error
    raise RuntimeError(f"Gemini judge call failed after {max_retries} attempts: {last_error}")



def call_foundry_anthropic_judge(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    row: dict[str, str],
    max_retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": deployment,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_prompt(row)}],
        "temperature": 0,
        "max_tokens": 500,
    }
    for attempt in range(max_retries):
        try:
            response = httpx.post(endpoint, headers=headers, json=payload, timeout=90)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code} {response.reason_phrase}: {response.text[:800]}"
                )
            data = response.json()
            content_blocks = data.get("content", [])
            text = "\n".join(
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
            return parse_json_lenient(text)
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"Claude judge call failed after {max_retries} attempts: {last_error}")


def normalize_label(value: Any, allowed: set[str], fallback: str) -> str:
    value = str(value or "").strip()
    return value if value in allowed else fallback


def normalize_judgment(obj: dict[str, Any]) -> dict[str, str]:
    return {
        "question_validity_clarity": normalize_label(
            obj.get("question_validity_clarity"), {"yes", "partial_unclear", "no"}, ""
        ),
        "question_fluency_naturalness": normalize_label(
            obj.get("question_fluency_naturalness"),
            {
                "mostly_natural_rephrased",
                "some_copied_acceptable",
                "too_much_copied_unnatural",
                "unclear",
            },
            "",
        ),
        "source_target_evidence_dependency": normalize_label(
            obj.get("source_target_evidence_dependency"),
            {
                "both_needed",
                "source_only_sufficient",
                "target_only_sufficient",
                "neither_mismatch_unclear",
            },
            "",
        ),
    }


def safe_model_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("_") or "llm_judge"


def default_output_path(input_path: Path, deployment: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.{safe_model_name(deployment)}{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LLM judge over a human annotation CSV.")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "foundry-anthropic", "azure-openai"],
        default=os.environ.get("HUMAN_ANNOTATION_LLM_PROVIDER", "foundry-anthropic"),
        help="LLM provider to use (default: foundry-anthropic)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ADGM_INPUT,
        help=f"Input human annotation CSV (default: {DEFAULT_ADGM_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output judged CSV. Default: input filename plus .<deployment>.csv",
    )
    parser.add_argument(
        "--deployment",
        default="",
        help="Deployment/model name. Defaults to Claude deployment for foundry-anthropic or AZURE_OPENAI_DEPLOYMENT_GPT52 for azure-openai.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name for --provider openai or gemini.",
    )
    parser.add_argument(
        "--endpoint",
        default="",
        help="Provider endpoint. For Claude, use the services.ai.azure.com /anthropic endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Provider API key. Prefer env vars instead of passing this on the command line.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a pilot run")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output CSV exists, keep completed rows and continue unfinished rows.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.0,
        help="Delay between successful calls, useful for free-tier rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=30,
        help="Maximum retry attempts per row. Useful for Gemini free-tier rate limits.",
    )
    args = parser.parse_args()

    if args.provider == "azure-openai" and not args.deployment:
        args.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT52", "") or os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT", ""
        )
    if args.provider == "openai":
        output_model_name = resolve_openai_model(args)
    elif args.provider == "gemini":
        output_model_name = resolve_gemini_model(args)
    else:
        output_model_name = args.deployment or "claude-opus-4-5-MBZUAI"
    output_path = args.output or default_output_path(args.input, output_model_name)

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if args.limit is not None:
        rows = rows[: args.limit]

    if args.provider == "openai":
        model = resolve_openai_model(args)
        print(f"OpenAI model: {model}")
        print(f"OpenAI API key set: {'yes' if (args.api_key or os.getenv('OPENAI_API_KEY')) else 'no'}")
        client = get_openai_client(args.api_key)
        anthropic_cfg = None
    elif args.provider == "gemini":
        model = resolve_gemini_model(args)
        gemini_api_key = get_gemini_api_key(args)
        print(f"Gemini model: {model}")
        print(f"Gemini API key set: yes")
        client = None
        anthropic_cfg = None
    elif args.provider == "azure-openai":
        preflight_azure_config(args.deployment)
        client = get_client()
        anthropic_cfg = None
    else:
        endpoint, api_key, deployment = get_foundry_anthropic_config(args)
        print(f"Foundry Anthropic endpoint: {endpoint}")
        print(f"Foundry Anthropic deployment: {deployment}")
        print(f"Foundry Anthropic API key set: {'yes' if api_key else 'no'}")
        client = None
        anthropic_cfg = (endpoint, api_key, deployment)
    out_fields = fieldnames + [c for c in ANNOTATION_COLS if c not in fieldnames]

    judged_rows: list[dict[str, str]] = []
    existing_by_id: dict[str, dict[str, str]] = {}
    if args.resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        existing_by_id = {
            r.get("qa_id") or r.get("annotation_id") or str(i): r
            for i, r in enumerate(existing_rows, start=1)
        }

    for i, row in enumerate(rows, start=1):
        row_id = row.get("qa_id") or row.get("annotation_id") or str(i)
        existing = existing_by_id.get(row_id)
        if existing and all(existing.get(c) for c in ANNOTATION_COLS[:3]):
            print(f"Skipping {i}/{len(rows)} already annotated: {row_id}")
            judged_rows.append(existing)
            continue

        print(f"Judging {i}/{len(rows)}: {row_id}")
        try:
            if args.provider == "openai":
                judgment = normalize_judgment(call_openai_judge(client, model, row))
            elif args.provider == "gemini":
                judgment = normalize_judgment(
                    call_gemini_judge(
                        api_key=gemini_api_key,
                        model=model,
                        row=row,
                        endpoint=args.endpoint,
                        max_retries=args.max_retries,
                    )
                )
            elif args.provider == "azure-openai":
                judgment = normalize_judgment(call_judge(client, args.deployment, row))
            else:
                endpoint, api_key, deployment = anthropic_cfg
                judgment = normalize_judgment(
                    call_foundry_anthropic_judge(
                        endpoint=endpoint,
                        api_key=api_key,
                        deployment=deployment,
                        row=row,
                    )
                )
        except ProviderQuotaExceeded as exc:
            write_rows(output_path, out_fields, judged_rows)
            print(f"Provider quota exhausted at row {i}/{len(rows)}: {row_id}")
            print(f"Saved {len(judged_rows)}/{len(rows)} rows to {output_path}")
            print("Rerun later with --resume to continue from the next unfinished row.")
            print(str(exc))
            return
        out_row = dict(row)
        out_row.update(judgment)
        out_row.setdefault("comment", "")
        judged_rows.append(out_row)
        write_rows(output_path, out_fields, judged_rows)
        print(f"Saved {len(judged_rows)}/{len(rows)} rows to {output_path}")
        if args.delay_s > 0 and i < len(rows):
            time.sleep(args.delay_s)

    write_rows(output_path, out_fields, judged_rows)

    print(f"Wrote {len(judged_rows)} annotated rows to {output_path}")


if __name__ == "__main__":
    main()
