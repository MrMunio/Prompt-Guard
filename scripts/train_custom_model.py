#!/usr/bin/env python3
# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0
"""
train_custom_model.py — Train a single custom SVM from a JSONL dataset.

Aligned with legacy train_l1_specialist.py pipeline:
  - Full L0 pre-processing (NFKC, HTML strip, zero-width removal, confusable replacement)
  - SHA-256 content deduplication before any split
  - Squash augmentation applied AFTER train/test split (eliminates data leakage)
  - L1 penalty (dual=False, class_weight="balanced") matching legacy LinearSVC config
  - Dynamic min_df: min_df=1 if training set < 50 records, else min_df=5

Reads --data-file (JSON lines: {"text": "...", "label": 0|1}) instead of YAML files.
Outputs --out-weights (path to .weights.json) instead of Rust codegen.
Reports JSON metrics to stdout: {
    "f1": float, "recall": float, "precision": float, "samples": int,
    "blend_samples": int, "dataset_blend_samples": int,
    "augmented_train_size": int, "test_size": int, "dedup_removed": int
}

Optional --blend-categories: blends all datasets for a canonical category into the training set.
Optional --blend-dataset-files: blends specific YAML files by absolute path.

Usage:
  python scripts/train_custom_model.py \\
    --data-file /tmp/model_uuid_train.jsonl \\
    --out-weights ./models/custom/uuid.weights.json

  # With category blending:
  python scripts/train_custom_model.py \\
    --data-file /tmp/model_uuid_train.jsonl \\
    --out-weights ./models/custom/uuid.weights.json \\
    --blend-categories instruction_override exfiltration \\
    --schema-dir ./schema/eval \\
    --cache-dir ./models/base_cache

  # With specific dataset file blending:
  python scripts/train_custom_model.py \\
    --data-file /tmp/model_uuid_train.jsonl \\
    --out-weights ./models/custom/uuid.weights.json \\
    --blend-dataset-files /abs/path/opensource_gandalf_attacks.yaml

  # Combining both:
  python scripts/train_custom_model.py \\
    --data-file /tmp/model_uuid_train.jsonl \\
    --out-weights ./models/custom/uuid.weights.json \\
    --blend-categories instruction_override \\
    --blend-dataset-files /abs/path/opensource_gandalf_attacks.yaml
"""

import argparse
import hashlib
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

# ---------------------------------------------------------------------------
# Minimum training-set size below which we relax min_df to avoid zero features.
# Below this threshold: min_df=1; at or above: min_df=5 (matches legacy pipeline).
# ---------------------------------------------------------------------------
MIN_DF_THRESHOLD = 50

INVALID_YAML_CTRL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)

# Zero-width / invisible chars (matches parapet L0 normalize::remove_zero_width)
ZERO_WIDTH_RE = re.compile(
    r"[\u200B-\u200D\u200E\u200F\u2060\u2061\u2062\u2063\uFEFF\u00AD]"
)

# ---------------------------------------------------------------------------
# Category → source YAML file mapping (mirrors train_base_models.py)
# ---------------------------------------------------------------------------

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

BENIGN_SOURCES: list[str] = [
    "opensource_no_robots_benign.yaml",
    "opensource_chatgpt_prompts_benign.yaml",
    "staging/opensource_notinject_benign.yaml",
    "staging/opensource_wildguardmix_benign.yaml",
    "l1_benign.yaml",
]

# ---------------------------------------------------------------------------
# L0 pre-processing
# Aligned with parapet runtime: NFKC normalization, HTML strip, zero-width
# removal, and YAML control character cleanup.
# ---------------------------------------------------------------------------

# Simple HTML tag stripper (no external deps required)
_HTML_TAG_RE = re.compile(r"<[^>]{0,200}>")
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{1,6}|#x[0-9a-fA-F]{1,6});")

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&apos;": "'", "&nbsp;": " ",
}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def apply_l0_transform(text: str) -> str:
    """
    Full L0 pre-processing pipeline matching the parapet runtime:
      1. Strip HTML tags and entities
      2. Remove zero-width / invisible characters
      3. NFKC Unicode normalization
      4. Strip YAML-invalid control characters
      5. Strip leading/trailing whitespace
    """
    text = strip_html(text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = INVALID_YAML_CTRL_RE.sub("", text)
    return text.strip()


def squash(text: str) -> str:
    """Mirror l1.rs::squash() — lowercase then keep only alphanumeric."""
    return "".join(c for c in unicodedata.normalize("NFC", text.lower()) if c.isalnum())


def content_hash(text: str) -> str:
    """SHA-256 hash of text for deduplication (matches train_l1_specialist.py dedup_entries)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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
                raw_text = str(obj.get("text", ""))
                text = apply_l0_transform(raw_text)
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
# Deduplication
# ---------------------------------------------------------------------------

def dedup_records(records: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
    """
    Deduplicate records by SHA-256 content hash (matches legacy dedup_entries).
    Returns (deduplicated_records, n_removed).
    """
    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for text, label in records:
        h = content_hash(text)
        if h not in seen:
            seen.add(h)
            unique.append((text, label))
    removed = len(records) - len(unique)
    return unique, removed


# ---------------------------------------------------------------------------
# YAML loading (only imported when blend is requested)
# ---------------------------------------------------------------------------

def _load_yaml_texts(path: Path, label: int) -> list[tuple[str, int]]:
    """Load texts from a YAML file. Applies L0 transform. Handles list-of-str and list-of-dict."""
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
            text = apply_l0_transform(item)
            if text:
                results.append((text, label))
        elif isinstance(item, dict):
            raw = item.get("text") or item.get("content") or item.get("prompt") or ""
            text = apply_l0_transform(str(raw))
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
        benign_records, _ = dedup_records(benign_records)
        write_jsonl(benign_cache, benign_records)
        print(f"  [blend] benign: cached {len(benign_records)} records → {benign_cache}", file=sys.stderr)

    for text, label in benign_records:
        h = content_hash(text)
        if h not in seen:
            seen.add(h)
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
            cat_records, _ = dedup_records(cat_records)
            write_jsonl(cat_cache, cat_records)
            print(f"  [blend] {cat}: cached {len(cat_records)} records → {cat_cache}", file=sys.stderr)

        for text, label in cat_records:
            h = content_hash(text)
            if h not in seen:
                seen.add(h)
                blended.append((text, label))

    return blended


def load_specific_datasets(
    file_paths: list[str],
    existing_seen: set[str] | None = None,
) -> list[tuple[str, int]]:
    """
    Load specific YAML dataset files by absolute path.
    Deduplicates against existing_seen (from category blend if both are used).
    Returns deduplicated (text, label) tuples.
    """
    seen: set[str] = set(existing_seen or set())
    records: list[tuple[str, int]] = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"  WARNING: blend-dataset-file not found: {file_path}", file=sys.stderr)
            continue
        try:
            import yaml
            try:
                loader = yaml.CSafeLoader
            except AttributeError:
                loader = yaml.SafeLoader
            with open(path, encoding="utf-8", errors="replace") as f:
                data = yaml.load(f, Loader=loader)
        except Exception as e:
            print(f"  WARNING: could not load {file_path}: {e}", file=sys.stderr)
            continue

        if not isinstance(data, list):
            print(f"  WARNING: {file_path} is not a list, skipping", file=sys.stderr)
            continue

        loaded = 0
        skipped = 0
        for item in data:
            if not isinstance(item, dict):
                skipped += 1
                continue
            raw = item.get("text") or item.get("content") or item.get("prompt") or ""
            text = apply_l0_transform(str(raw))
            if not text:
                skipped += 1
                continue

            raw_label = item.get("label")
            if raw_label is None:
                skipped += 1
                continue
            if raw_label in ("malicious", 1, True):
                label = 1
            elif raw_label in ("benign", 0, False):
                label = 0
            else:
                skipped += 1
                continue

            h = content_hash(text)
            if h not in seen:
                seen.add(h)
                records.append((text, label))
                loaded += 1

        attacks_in_file = sum(1 for _, l in records[-loaded:] if l == 1)
        benign_in_file = loaded - attacks_in_file
        print(
            f"  [blend-dataset] {path.name}: {loaded} records loaded "
            f"({attacks_in_file} attack, {benign_in_file} benign), {skipped} skipped",
            file=sys.stderr,
        )

    return records


# ---------------------------------------------------------------------------
# Dynamic min_df selection
# ---------------------------------------------------------------------------

def select_min_df(n_samples: int) -> int:
    """
    Select min_df based on training set size:
      - n_samples < MIN_DF_THRESHOLD (50): min_df=1  (small dataset, keep all features)
      - n_samples >= MIN_DF_THRESHOLD (50): min_df=5  (matches legacy train_l1_specialist.py)
    """
    if n_samples < MIN_DF_THRESHOLD:
        print(
            f"  [min_df] training set size={n_samples} < {MIN_DF_THRESHOLD} → using min_df=1",
            file=sys.stderr,
        )
        return 1
    print(
        f"  [min_df] training set size={n_samples} >= {MIN_DF_THRESHOLD} → using min_df=5 (legacy default)",
        file=sys.stderr,
    )
    return 5


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
    parser.add_argument("--blend-dataset-files", nargs="*", default=[],
                        help="Specific dataset YAML files to blend in by absolute path")
    parser.add_argument("--schema-dir",         default="./schema/eval",
                        help="Path to schema/eval directory containing base YAML files")
    parser.add_argument("--cache-dir",          default="./models/base_cache",
                        help="Directory to store per-category JSONL blend caches")
    args = parser.parse_args()

    # --- Load client + mirror records (L0 applied inside load_jsonl) ---
    client_records = load_jsonl(args.data_file)
    print(f"  [data] loaded {len(client_records)} client+mirror records from {args.data_file}", file=sys.stderr)

    # --- Category-level blend ---
    blend_records: list[tuple[str, int]] = []
    if args.blend_categories:
        blend_records = load_base_blend(
            categories=args.blend_categories,
            schema_dir=Path(args.schema_dir),
            cache_dir=Path(args.cache_dir),
        )
        print(f"  [blend] total category-blended records: {len(blend_records)}", file=sys.stderr)

    # --- Specific dataset-level blend ---
    dataset_blend_records: list[tuple[str, int]] = []
    if args.blend_dataset_files:
        already_seen = {content_hash(text) for text, _ in blend_records}
        dataset_blend_records = load_specific_datasets(
            file_paths=args.blend_dataset_files,
            existing_seen=already_seen,
        )
        print(
            f"  [blend-dataset] total dataset-blended records: {len(dataset_blend_records)}",
            file=sys.stderr,
        )

    # --- Combine: client records + category blend + specific dataset blend ---
    all_records_raw = client_records + blend_records + dataset_blend_records

    # --- Global deduplication (by content hash) ---
    all_records, dedup_removed = dedup_records(all_records_raw)
    if dedup_removed:
        print(f"  [dedup] removed {dedup_removed} duplicate records", file=sys.stderr)

    if len(all_records) < 4:
        json.dump({"error": "Insufficient training data (minimum 4 records required after deduplication)"}, sys.stdout)
        sys.exit(1)

    texts  = [t for t, _ in all_records]
    labels = [l for _, l in all_records]

    # --- Stratified train/holdout split (on ORIGINAL records, NO squash yet) ---
    # Squash augmentation is applied AFTER splitting to prevent data leakage.
    # (Leakage path: original in X_train, its squashed clone in X_test → F1=1.0)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=42
        )
    except ValueError:
        # Not enough samples for stratified split — fall back to random.
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.15, random_state=42
        )

    # --- Squash augmentation — applied to TRAIN ONLY ---
    # Mirror: L1 runtime uses both raw + squashed text paths; max(raw, squashed) score.
    # Augment train set with squashed variants to match runtime behaviour.
    X_train_aug = X_train + [squash(t) for t in X_train]
    y_train_aug = y_train + y_train

    # --- Dynamic min_df selection (based on pre-augmentation train size) ---
    min_df = select_min_df(len(X_train))

    ngram_range = (args.ngram_min, args.ngram_max)
    vec = CountVectorizer(
        analyzer=args.analyzer,
        ngram_range=ngram_range,
        max_features=args.max_features,
        binary=True,
        min_df=min_df,
    )
    X_tr = vec.fit_transform(X_train_aug)
    X_te = vec.transform(X_test)

    # --- LinearSVC with L1 penalty (matches legacy train_l1_specialist.py) ---
    # penalty="l1", dual=False, class_weight="balanced" → sparse features,
    # balanced class weighting for imbalanced datasets.
    clf = LinearSVC(
        penalty="l1",
        dual=False,
        C=args.c,
        class_weight="balanced",
        max_iter=2000,
    )
    clf.fit(X_tr, y_train_aug)

    preds = clf.predict(X_te)
    f1  = float(f1_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    pre = float(precision_score(y_test, preds, zero_division=0))

    print(f"  [metrics] F1={f1:.4f}  Recall={rec:.4f}  Precision={pre:.4f}", file=sys.stderr)
    print(f"  [metrics] train_aug={len(X_train_aug)}  test={len(X_test)}", file=sys.stderr)

    # --- Emit weights.json ---
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

    # --- Report metrics to stdout (read by train.rs) ---
    # samples        = client records only (records in the submitted JSONL / DB export)
    # blend_samples  = category-level base corpus records blended in
    # dataset_blend_samples = specific dataset-level blend records
    # total_samples  = all records after dedup — what the model was actually trained on
    #                  (client + blend + dataset_blend, before augmentation)
    total_samples = len(all_records)  # post-dedup, pre-augmentation
    metrics = {
        "f1": f1,
        "recall": rec,
        "precision": pre,
        "samples": len(client_records),
        "blend_samples": len(blend_records),
        "dataset_blend_samples": len(dataset_blend_records),
        "total_samples": total_samples,
        "augmented_train_size": len(X_train_aug),
        "test_size": len(X_test),
        "dedup_removed": dedup_removed,
        "min_df_used": min_df,
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
