#!/usr/bin/env python3
# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0
"""
mirror_augment.py — Mirror dataset augmentation via LLM.

For each client training record stored in the DB for a given model,
generates one mirror counterpart (attack→benign or benign→attack)
and inserts it into training_records (source='mirror_generated').

Based on the Mirror Design Pattern (arxiv 2603.11875v1 §4.2).

Usage (called by train.rs background task):
  python scripts/mirror_augment.py \
    --model-id <uuid> \
    --db-url sqlite:guardrail.db \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --api-key sk-...
"""

import argparse
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mirror system prompt (from Mirror Design Pattern paper)
# ---------------------------------------------------------------------------

MIRROR_SYSTEM_PROMPT = """You are a data augmentation assistant for a prompt injection classifier training dataset.

The Mirror Design Pattern pairs each attack example with a benign "mirror" counterpart
that shares the same language, topic, approximate length, and format — but does NOT
attempt to override model instructions, reassign roles, exfiltrate data, or otherwise
hijack model behavior.

For each input record, generate ONE mirror counterpart:
- If the input is an ATTACK (label=1): generate a BENIGN text (label=0) on the same topic,
  same approximate length, same language, phrased as a normal helpful user request.
- If the input is BENIGN (label=0): generate an ATTACK text (label=1) that mimics the
  same topic/length/language but attempts instruction override or prompt injection.

Respond with ONLY valid JSON: {"text": "...", "label": 0 or 1}
No explanation, no markdown, no extra fields."""


def call_llm(client: httpx.Client, base_url: str, model: str, api_key: str,
             text: str, label: int, max_retries: int = 3) -> dict | None:
    """Call the LLM to generate a mirror record. Returns {"text": str, "label": int} or None."""
    user_msg = json.dumps({"text": text, "label": label})
    for attempt in range(max_retries):
        try:
            resp = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": MIRROR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)
            if "text" in parsed and "label" in parsed and parsed["label"] in (0, 1):
                return {"text": str(parsed["text"]), "label": int(parsed["label"])}
        except (json.JSONDecodeError, KeyError, httpx.HTTPError) as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}", file=sys.stderr)
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_client_records(conn: sqlite3.Connection, model_id: str) -> list[dict]:
    """Fetch client records that don't yet have a mirror counterpart."""
    cur = conn.execute(
        """SELECT id, text, label FROM training_records
           WHERE model_id = ? AND source = 'client'
           AND id NOT IN (SELECT mirror_of FROM training_records WHERE mirror_of IS NOT NULL)""",
        (model_id,)
    )
    return [{"id": row[0], "text": row[1], "label": row[2]} for row in cur.fetchall()]


def insert_mirror_record(conn: sqlite3.Connection, model_id: str,
                          text: str, label: int, mirror_of: str, now: str):
    record_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO training_records (id, model_id, text, label, source, mirror_of, created_at)
           VALUES (?, ?, ?, ?, 'mirror_generated', ?, ?)""",
        (record_id, model_id, text, label, mirror_of, now)
    )


def main():
    parser = argparse.ArgumentParser(description="Mirror augmentation via LLM")
    parser.add_argument("--model-id",  required=True)
    parser.add_argument("--db-url",    required=True, help="sqlite:path or 'sqlite' for default")
    parser.add_argument("--base-url",  required=True)
    parser.add_argument("--model",     required=True)
    parser.add_argument("--api-key",   required=True)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    # Resolve DB path from DATABASE_URL style value.
    db_path = args.db_url.replace("sqlite:", "").replace("sqlite://", "")
    if not db_path or db_path == "sqlite":
        db_path = "guardrail.db"

    conn = sqlite3.connect(db_path)
    records = fetch_client_records(conn, args.model_id)
    print(f"Found {len(records)} client records to augment", file=sys.stderr)

    if not records:
        print("No records to augment.", file=sys.stderr)
        conn.close()
        return

    if args.dry_run:
        print(f"[dry-run] would generate {len(records)} mirror records", file=sys.stderr)
        conn.close()
        return

    client = httpx.Client()
    now = datetime.now(timezone.utc).isoformat()
    success = 0
    for rec in records:
        mirror = call_llm(client, args.base_url, args.model, args.api_key,
                          rec["text"], rec["label"])
        if mirror:
            insert_mirror_record(conn, args.model_id, mirror["text"], mirror["label"],
                                  rec["id"], now)
            success += 1
        else:
            print(f"  WARNING: Could not generate mirror for record {rec['id']}", file=sys.stderr)

    conn.commit()
    conn.close()
    client.close()
    print(f"Mirror augmentation complete: {success}/{len(records)} records generated", file=sys.stderr)


if __name__ == "__main__":
    main()
