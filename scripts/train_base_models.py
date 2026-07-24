#!/usr/bin/env python3
# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0
"""
train_base_models.py — Trains all 9 base SVM models and emits .weights.json files.

Models trained:
  allrounder           — trained on ALL 8 category datasets combined
  instruction_override
  roleplay_jailbreak
  meta_probe
  exfiltration
  adversarial_suffix
  indirect_injection
  obfuscation
  constraint_bypass

Output:
  <models_dir>/base/<name>.weights.json

Each .weights.json has the format:
  { "bias": float, "weights": { "<ngram>": float, ... }, "analyzer": str, "ngram_range": [int, int] }

Usage:
  python scripts/train_base_models.py --models-dir ./models

Source data: schema/eval/ YAML files (same corpus used by train_l1_specialist.py).
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

# ---------------------------------------------------------------------------
# Category → source file mapping
# Source files are the same YAML attack datasets used by train_l1_specialist.py
# ---------------------------------------------------------------------------

# Locate dataset schema directory inside parapet-guardrail
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_SCHEMA_DIRS = [
    SCRIPT_DIR.parent / "schema",
    SCRIPT_DIR.parent / "schema" / "eval",
    Path("schema"),
    Path("schema/eval"),
]
SCHEMA_DIR = next((p for p in CANDIDATE_SCHEMA_DIRS if (p / "opensource_no_robots_benign.yaml").exists() or (p / "l1_benign.yaml").exists()), SCRIPT_DIR.parent / "schema")

# Full canonical dataset mapping (same exact sources as train_l1_specialist.py)
ATTACK_SOURCES = {
    "instruction_override": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "staging/opensource_notinject_benign.yaml",
        "l1_attacks.yaml",
    ],
    "roleplay_jailbreak": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "opensource_jailbreak_cls_attacks.yaml",
        "opensource_hackaprompt_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "meta_probe": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "exfiltration": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "opensource_jailbreak_cls_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "adversarial_suffix": [
        "opensource_jailbreak_cls_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "indirect_injection": [
        "opensource_jailbreak_cls_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "obfuscation": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "opensource_jailbreak_cls_attacks.yaml",
        "opensource_hackaprompt_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "constraint_bypass": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "opensource_jailbreak_cls_attacks.yaml",
        "opensource_hackaprompt_attacks.yaml",
        "l1_attacks.yaml",
    ],
}

BENIGN_SOURCES = [
    "opensource_no_robots_benign.yaml",
    "opensource_chatgpt_prompts_benign.yaml",
    "staging/opensource_notinject_benign.yaml",
    "staging/opensource_wildguardmix_benign.yaml",
    "l1_benign.yaml",
]

FETCH_SCRIPT_MAP = {
    "opensource_no_robots_benign.yaml": "scripts/sources/fetch_no_robots.py",
    "opensource_chatgpt_prompts_benign.yaml": "scripts/sources/fetch_chatgpt_prompts.py",
    "staging/opensource_notinject_benign.yaml": "scripts/sources/fetch_notinject.py",
    "staging/opensource_wildguardmix_benign.yaml": "scripts/sources/fetch_wildguardmix.py",
    "opensource_chatgpt_jailbreak_attacks.yaml": "scripts/sources/fetch_chatgpt_jailbreak.py",
    "opensource_jailbreak_cls_attacks.yaml": "scripts/sources/fetch_jailbreak_cls.py",
    "opensource_hackaprompt_attacks.yaml": "scripts/sources/fetch_hackaprompt.py",
}


def ensure_dataset_files():
    """Auto-download missing raw training dataset files if not present."""
    import subprocess
    for filename, script_path in FETCH_SCRIPT_MAP.items():
        target_path = SCHEMA_DIR / filename
        if not target_path.exists():
            script_full = SCRIPT_DIR.parent / script_path
            if script_full.exists():
                print(f"Dataset {filename} missing. Triggering auto-fetch via {script_path}...", file=sys.stderr)
                try:
                    subprocess.run([sys.executable, str(script_full)], check=True)
                except Exception as e:
                    print(f"Warning: Failed to auto-fetch {filename}: {e}", file=sys.stderr)

# Analyzer config per specialist (mirrors parapet L1 design).
SPECIALIST_CONFIGS = {
    "allrounder":          {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 15000},
    "instruction_override":{"analyzer": "word",    "ngram_range": (1, 3), "max_features": 10000},
    "roleplay_jailbreak":  {"analyzer": "word",    "ngram_range": (2, 4), "max_features": 10000},
    "meta_probe":          {"analyzer": "word",    "ngram_range": (1, 2), "max_features": 8000},
    "exfiltration":        {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 10000},
    "adversarial_suffix":  {"analyzer": "char",    "ngram_range": (3, 5), "max_features": 10000},
    "indirect_injection":  {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 10000},
    "obfuscation":         {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 10000},
    "constraint_bypass":   {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 10000},
}

# ---------------------------------------------------------------------------
# Helpers (from train_l1_specialist.py)
# ---------------------------------------------------------------------------

INVALID_YAML_CTRL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def strip_invalid_yaml_controls(text: str) -> str:
    return INVALID_YAML_CTRL_RE.sub("", text)


def squash(text: str) -> str:
    """Mirror l1.rs::squash() — lowercase then keep only alphanumeric."""
    return "".join(c for c in unicodedata.normalize("NFC", text.lower()) if c.isalnum())


def load_yaml_texts(path: Path, label: int) -> list[tuple[str, int]]:
    """Load texts + labels from a YAML file. Handles list-of-str and list-of-dict."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        data = yaml.load(f, Loader=YAML_LOADER)
    if not isinstance(data, list):
        return []
    results = []
    for item in data:
        if isinstance(item, str):
            text = strip_invalid_yaml_controls(item).strip()
            if text:
                results.append((text, label))
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("prompt") or ""
            text = strip_invalid_yaml_controls(str(text)).strip()
            if text:
                results.append((text, label))
    return results


def load_attacks(category: str) -> list[tuple[str, int]]:
    sources = ATTACK_SOURCES.get(category, list(ATTACK_SOURCES.values())[0])
    records = []
    for src in sources:
        records.extend(load_yaml_texts(SCHEMA_DIR / src, 1))
    # Dedup by content.
    seen = set()
    unique = []
    for text, label in records:
        if text not in seen:
            seen.add(text)
            unique.append((text, label))
    return unique


def load_benign() -> list[tuple[str, int]]:
    records = []
    for src in BENIGN_SOURCES:
        records.extend(load_yaml_texts(SCHEMA_DIR / src, 0))
    seen = set()
    unique = []
    for text, label in records:
        if text not in seen:
            seen.add(text)
            unique.append((text, label))
    return unique


def train_and_emit(name: str, texts: list[str], labels: list[int],
                   cfg: dict, out_path: Path, dry_run: bool = False) -> dict:
    """Train LinearSVC and emit .weights.json. Returns metrics."""
    texts_aug = texts + [squash(t) for t in texts]
    labels_aug = labels + labels

    X_train, X_test, y_train, y_test = train_test_split(
        texts_aug, labels_aug, test_size=0.15, stratify=labels_aug, random_state=42
    )

    vec = CountVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        max_features=cfg["max_features"],
        binary=True,
        min_df=2,
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    if dry_run:
        print(f"  [dry-run] {name}: {len(texts)} samples, skipping fit")
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "samples": len(texts)}

    clf = LinearSVC(C=1.0, max_iter=2000)
    clf.fit(X_tr, y_train)

    from sklearn.metrics import classification_report
    preds = clf.predict(X_te)
    from sklearn.metrics import f1_score, recall_score, precision_score
    f1  = f1_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    pre = precision_score(y_test, preds, zero_division=0)

    # Emit weights.json.
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    weights = {name: float(w) for name, w in zip(feature_names, coef) if w != 0.0}
    bias = float(clf.intercept_[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "bias": bias,
            "weights": weights,
            "analyzer": cfg["analyzer"],
            "ngram_range": list(cfg["ngram_range"]),
        }, f, ensure_ascii=False)

    print(f"  {name}: F1={f1:.3f}  Recall={rec:.3f}  Precision={pre:.3f}  samples={len(texts)}")
    return {"f1": f1, "recall": rec, "precision": pre, "samples": len(texts)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train all 9 base SVM models")
    parser.add_argument("--models-dir", default="./models",
                        help="Directory to write base/*.weights.json files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs without training")
    args = parser.parse_args()

    models_dir = Path(args.models_dir) / "base"
    models_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset_files()

    print("Loading benign data...")
    benign = load_benign()
    if not benign:
        print("WARNING: No benign data found. Check BENIGN_SOURCES paths.", file=sys.stderr)

    all_attacks = []
    specialists = [k for k in SPECIALIST_CONFIGS if k != "allrounder"]

    for category in specialists:
        out_path = models_dir / f"{category}.weights.json"
        if out_path.exists() and not args.dry_run:
            print(f"  {category}: cache hit, skipping")
            continue

        print(f"Training {category}...")
        attacks = load_attacks(category)
        all_attacks.extend(attacks)
        if not attacks:
            print(f"  WARNING: no attack data for {category}", file=sys.stderr)
            continue

        # Balance classes.
        n = min(len(attacks), len(benign))
        paired_attacks = attacks[:n]
        paired_benign = benign[:n]
        texts  = [t for t, _ in paired_attacks + paired_benign]
        labels = [l for _, l in paired_attacks + paired_benign]

        train_and_emit(category, texts, labels, SPECIALIST_CONFIGS[category],
                       out_path, dry_run=args.dry_run)

    # Allrounder — trained on ALL attack data combined.
    out_path = models_dir / "allrounder.weights.json"
    if not out_path.exists() or args.dry_run:
        print("Training allrounder (all categories combined)...")
        seen = set()
        unique_attacks = []
        for text, label in all_attacks:
            if text not in seen:
                seen.add(text)
                unique_attacks.append((text, label))

        n = min(len(unique_attacks), len(benign))
        texts  = [t for t, _ in unique_attacks[:n] + benign[:n]]
        labels = [l for _, l in unique_attacks[:n] + benign[:n]]
        train_and_emit("allrounder", texts, labels, SPECIALIST_CONFIGS["allrounder"],
                       out_path, dry_run=args.dry_run)
    else:
        print("  allrounder: cache hit, skipping")

    print("Done.")


if __name__ == "__main__":
    main()
