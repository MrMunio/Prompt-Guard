#!/usr/bin/env python3
# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0
"""
generate_regex.py — Generate regex patterns from plain-text descriptions via LLM.

Called by patterns.rs when a user submits a non-regex input string.

Usage:
  python scripts/generate_regex.py \
    --description "detect when user asks to reveal the system prompt" \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --api-key sk-...

Outputs JSON to stdout: {"patterns": ["regex1", "regex2"]}
"""

import argparse
import json
import re
import sys
import time

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a regex pattern assistant. The user will describe, in plain English, 
text patterns they want to detect in user input to an AI assistant.

Generate one or more Python-compatible regular expressions that satisfy the user's intent.
Rules:
- Prefer simple, readable patterns over complex ones
- Use case-insensitive matching where appropriate (use (?i) flag or note that matching is case-insensitive)
- Cover the main variations the user described
- Do NOT generate patterns that are too broad (e.g. matching everything)
- Generate between 1 and 5 patterns

Respond with ONLY valid JSON: {"patterns": ["regex1", "regex2", ...]}
No explanation, no markdown, no extra fields."""


def call_llm(base_url: str, model: str, api_key: str,
             description: str, max_retries: int = 3) -> list[str]:
    client = httpx.Client()
    for attempt in range(max_retries):
        try:
            resp = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": description},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)
            patterns = parsed.get("patterns", [])
            if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
                # Validate each pattern compiles.
                valid = []
                for p in patterns:
                    try:
                        re.compile(p)
                        valid.append(p)
                    except re.error as e:
                        print(f"  WARNING: LLM generated invalid regex '{p}': {e}", file=sys.stderr)
                if valid:
                    client.close()
                    return valid
        except (json.JSONDecodeError, KeyError, httpx.HTTPError) as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}", file=sys.stderr)
            time.sleep(1.0 * (attempt + 1))
    client.close()
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate regex patterns from plain-text description via LLM")
    parser.add_argument("--description", required=True, help="Plain-text description of patterns to detect")
    parser.add_argument("--base-url",    required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--api-key",     required=True)
    args = parser.parse_args()

    patterns = call_llm(args.base_url, args.model, args.api_key, args.description)
    if not patterns:
        # Fallback: emit a word-boundary pattern from the description.
        words = re.findall(r'\w+', args.description.lower())
        if words:
            fallback = r'(?i)\b(' + '|'.join(re.escape(w) for w in words[:5]) + r')\b'
            patterns = [fallback]
            print(f"  WARNING: LLM failed, using fallback pattern: {fallback}", file=sys.stderr)
        else:
            print(json.dumps({"error": "Could not generate any patterns"}))
            sys.exit(1)

    print(json.dumps({"patterns": patterns}))


if __name__ == "__main__":
    main()
