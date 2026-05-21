#!/usr/bin/env python3
"""Minimal Azure AI Foundry Claude connectivity test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(".env"), override=False)
except Exception:
    pass


def main() -> int:
    endpoint = (
        os.getenv("AZURE_AI_FOUNDRY_ANTHROPIC_ENDPOINT")
        or "https://claude-fsra-resource.services.ai.azure.com/anthropic/v1/messages"
    ).strip()
    if endpoint.endswith("/anthropic"):
        endpoint = f"{endpoint}/v1/messages"
    elif not endpoint.endswith("/v1/messages"):
        endpoint = f"{endpoint.rstrip('/')}/anthropic/v1/messages"

    deployment = (
        os.getenv("AZURE_AI_FOUNDRY_ANTHROPIC_DEPLOYMENT")
        or "claude-opus-4-5-MBZUAI"
    ).strip()
    api_key = (
        os.getenv("AZURE_AI_FOUNDRY_API_KEY")
        or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
        or os.getenv("AZURE_API_KEY")
        or ""
    ).strip()

    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API key set: {'yes' if api_key else 'no'}")
    if not api_key:
        print("Missing key. Set AZURE_AI_FOUNDRY_API_KEY.")
        return 1

    payload = {
        "model": deployment,
        "messages": [{"role": "user", "content": 'Return ONLY JSON: {"ok": true}'}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    response = httpx.post(endpoint, headers=headers, json=payload, timeout=90)
    print(f"Status: {response.status_code}")
    print("Body:")
    try:
        print(json.dumps(response.json(), indent=2)[:2000])
    except Exception:
        print(response.text[:2000])
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
