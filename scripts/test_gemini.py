#!/usr/bin/env python3
"""Minimal Gemini API connectivity test."""

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
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base}/v1beta/models/{model}:generateContent"

    print(f"Model: {model}")
    print(f"URL: {url}")
    print(f"API key set: {'yes' if api_key else 'no'}")
    if not api_key:
        print("Missing key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        return 1

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": 'Return exactly this JSON: {"ok": true}'}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
            "thinkingConfig": {
                "thinkingLevel": "minimal",
            },
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=90)
    print(f"Status: {response.status_code}")
    print("Body:")
    try:
        print(json.dumps(response.json(), indent=2)[:3000])
    except Exception:
        print(response.text[:3000])
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
