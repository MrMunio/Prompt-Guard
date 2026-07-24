#!/usr/bin/env python3
# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0
"""
train_custom_model.py — Train a single custom SVM from a JSONL dataset.

Adapted from train_l1_specialist.py with these changes:
  - Reads --data-file (JSON lines: {"text": "...", "label": 0|1}) instead of YAML files
  - Outputs --out-weights (path to .weights.json) instead of Rust codegen
  - Reports JSON metrics to stdout: {"f1": float, "recall": float, "precision": float, "samples": int}
  - Optional --blend-categories: blends base corpus attack+benign data into the training set.
    Base YAML files are read from --schema-dir and cached as JSONL under --cache-dir.
    Cached JSONL files are reused on subsequent calls — no re-parsing of large YAML files.

Usage:
  python scripts/train_custom_model.py \\
    --data-file /tmp/model_uuid_train.jsonl \\
    --out-weights ./models/custom/uuid.weights.json

  # With base corpus blending:
  python scripts/train_custom_model.py \\
    --data-file /tmp/model_uuid_train.jsonl \\
    --out-weights ./models/custom/uuid.weights.json \\
    --blend-categories instruction_override exfiltration \\
    --schema-dir ./schema/eval \\
    --cache-dir ./models/base_cache
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

INVALID_YAML_CTRL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)

# ---------------------------------------------------------------------------
# Category → source YAML file mapping (mirrors train_base_models.py)
# ---------------------------------------------------------------------------

# All known attack source files per category.
ATTACK_SOURCES: dict[str, list[str]] = {
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

# Benign sources shared across all categories.
BENIGN_SOURCES: list[str] = [
    "opensource_no_robots_benign.yaml",
    "opensource_chatgpt_prompts_benign.yaml",
    "staging/opensource_notinject_benign.yaml",
    "staging/opensource_wildguardmix_benign.yaml",
    "l1_benign.yaml",
]

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    return INVALID_YAML_CTRL_RE.sub("", text).strip()


def squash(text: str) -> str:
    """Mirror l1.rs::squash() — lowercase then keep only alphanumeric."""
    return "".join(c for c in unicodedata.normalize("NFC", text.lower()) if c.isalnum())


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[tuple[str, int]]:
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = clean(str(obj.get("text", "")))
                label = int(obj.get("label", -1))
                if text and label in (0, 1):
                    records.append((text, label))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARNING: line {line_no} skipped: {e}", file=sys.stderr)
    return records


def write_jsonl(path: Path, records: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, label in records:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# YAML loading (only imported when blend is requested)
# ---------------------------------------------------------------------------

def _load_yaml_texts(path: Path, label: int) -> list[tuple[str, int]]:
    """Load texts from a YAML file. Handles list-of-str and list-of-dict."""
    if not path.exists():
        return []
    try:
        import yaml
        try:
            loader = yaml.CSafeLoader
        except AttributeError:
            loader = yaml.SafeLoader
        with open(path, encoding="utf-8", errors="replace") as f:
            data = yaml.load(f, Loader=loader)
    except Exception as e:
        print(f"  WARNING: could not load YAML {path}: {e}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    results = []
    for item in data:
        if isinstance(item, str):
            text = clean(item)
            if text:
                results.append((text, label))
        elif isinstance(item, dict):
            raw = item.get("text") or item.get("content") or item.get("prompt") or ""
            text = clean(str(raw))
            if text:
                results.append((text, label))
    return results


# ---------------------------------------------------------------------------
# Base corpus blend with JSONL caching
# ---------------------------------------------------------------------------

def load_base_blend(
    categories: list[str],
    schema_dir: Path,
    cache_dir: Path,
) -> list[tuple[str, int]]:
    """
    Load base corpus records for the given categories. Uses per-category JSONL cache
    under cache_dir so YAML is only parsed once. Falls back to live YAML parsing
    when cache is missing, then writes the cache for next time.

    Returns a deduplicated list of (text, label) tuples.
    """
    blended: list[tuple[str, int]] = []
    seen: set[str] = set()

    # Resolve "all" shorthand.
    if categories == ["all"]:
        categories = list(ATTACK_SOURCES.keys())

    # ----- benign cache -----
    benign_cache = cache_dir / "_benign.jsonl"
    if benign_cache.exists():
        benign_records = load_jsonl(str(benign_cache))
        print(f"  [blend] benign: cache hit ({len(benign_records)} records)", file=sys.stderr)
    else:
        print(f"  [blend] benign: cache miss — loading from YAML...", file=sys.stderr)
        benign_records = []
        for src in BENIGN_SOURCES:
            benign_records.extend(_load_yaml_texts(schema_dir / src, 0))
        # Dedup.
        seen_b: set[str] = set()
        unique_benign: list[tuple[str, int]] = []
        for text, label in benign_records:
            if text not in seen_b:
                seen_b.add(text)
                unique_benign.append((text, label))
        benign_records = unique_benign
        write_jsonl(benign_cache, benign_records)
        print(f"  [blend] benign: cached {len(benign_records)} records → {benign_cache}", file=sys.stderr)

    for text, label in benign_records:
        if text not in seen:
            seen.add(text)
            blended.append((text, label))

    # ----- per-category attack cache -----
    for cat in categories:
        if cat not in ATTACK_SOURCES:
            print(f"  WARNING: unknown blend category '{cat}', skipping", file=sys.stderr)
            continue

        cat_cache = cache_dir / f"{cat}.jsonl"
        if cat_cache.exists():
            cat_records = load_jsonl(str(cat_cache))
            print(f"  [blend] {cat}: cache hit ({len(cat_records)} records)", file=sys.stderr)
        else:
            print(f"  [blend] {cat}: cache miss — loading from YAML...", file=sys.stderr)
            cat_records = []
            for src in ATTACK_SOURCES[cat]:
                cat_records.extend(_load_yaml_texts(schema_dir / src, 1))
            # Dedup.
            seen_c: set[str] = set()
            unique_cat: list[tuple[str, int]] = []
            for text, label in cat_records:
                if text not in seen_c:
                    seen_c.add(text)
                    unique_cat.append((text, label))
            cat_records = unique_cat
            write_jsonl(cat_cache, cat_records)
            print(f"  [blend] {cat}: cached {len(cat_records)} records → {cat_cache}", file=sys.stderr)

        for text, label in cat_records:
            if text not in seen:
                seen.add(text)
                blended.append((text, label))

    return blended


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a custom SVM from a JSONL dataset")
    parser.add_argument("--data-file",         required=True,  help="Path to JSONL training data (client + mirror records)")
    parser.add_argument("--out-weights",        required=True,  help="Path to write .weights.json")
    parser.add_argument("--analyzer",           default="char_wb", choices=["char_wb", "char", "word"],
                        help="N-gram analyzer (default: char_wb)")
    parser.add_argument("--ngram-min",          type=int, default=3)
    parser.add_argument("--ngram-max",          type=int, default=5)
    parser.add_argument("--max-features",       type=int, default=15000)
    parser.add_argument("--c",                  type=float, default=1.0, help="LinearSVC C")
    # Blend arguments
    parser.add_argument("--blend-categories",   nargs="*", default=[],
                        help="Base corpus categories to blend in (e.g. instruction_override exfiltration, or 'all')")
    parser.add_argument("--schema-dir",         default="./schema/eval",
                        help="Path to schema/eval directory containing base YAML files")
    parser.add_argument("--cache-dir",          default="./models/base_cache",
                        help="Directory to store per-category JSONL blend caches")
    args = parser.parse_args()

    # Load client + mirror records.
    client_records = load_jsonl(args.data_file)

    # Load and append blend records.
    blend_records: list[tuple[str, int]] = []
    if args.blend_categories:
        blend_records = load_base_blend(
            categories=args.blend_categories,
            schema_dir=Path(args.schema_dir),
            cache_dir=Path(args.cache_dir),
        )
        print(f"  [blend] total blended records: {len(blend_records)}", file=sys.stderr)

    # Combine: client records take precedence, blend provides additional context.
    all_records = client_records + blend_records

    if len(all_records) < 4:
        json.dump({"error": "Insufficient training data (minimum 4 records required)"}, sys.stdout)
        sys.exit(1)

    texts  = [t for t, _ in all_records]
    labels = [l for _, l in all_records]

    # Squash augmentation — mirrors L1 training pipeline.
    texts_aug  = texts + [squash(t) for t in texts]
    labels_aug = labels + labels

    # Stratified train/holdout split.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts_aug, labels_aug, test_size=0.15, stratify=labels_aug, random_state=42
        )
    except ValueError:
        # Not enough samples for stratified split — use random.
        X_train, X_test, y_train, y_test = train_test_split(
            texts_aug, labels_aug, test_size=0.15, random_state=42
        )

    ngram_range = (args.ngram_min, args.ngram_max)
    vec = CountVectorizer(
        analyzer=args.analyzer,
        ngram_range=ngram_range,
        max_features=args.max_features,
        binary=True,
        min_df=1,
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    clf = LinearSVC(C=args.c, max_iter=2000)
    clf.fit(X_tr, y_train)

    preds = clf.predict(X_te)
    f1  = float(f1_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    pre = float(precision_score(y_test, preds, zero_division=0))

    # Emit weights.json.
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    weights = {str(name): float(w) for name, w in zip(feature_names, coef) if w != 0.0}
    bias = float(clf.intercept_[0])

    out_path = Path(args.out_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "bias": bias,
            "weights": weights,
            "analyzer": args.analyzer,
            "ngram_range": list(ngram_range),
        }, f, ensure_ascii=False)

    # Report metrics to stdout (read by train.rs).
    # samples = client records only (blend corpus is background signal, not reported as "user samples").
    metrics = {
        "f1": f1,
        "recall": rec,
        "precision": pre,
        "samples": len(client_records),
        "blend_samples": len(blend_records),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
