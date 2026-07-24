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

Hyperparameters are aligned with the legacy train_l1_specialist.py defaults:
  - analyzer:       char_wb  (all specialists — matches paper §4 design)
  - ngram_range:    (3, 5)
  - max_features:   25000
  - min_df:         5
  - binary:         True     (runtime scoring is additive, no TF-IDF normalization)
  - C:              0.1      (regularisation — matches legacy default)
  - class_weight:   balanced (handles attack/benign imbalance)
  - max_iter:       100000
  - holdout_pct:    0.2      (80/20 train/holdout split)
  - seed:           42
  - squash_augment: True     (doubles train data with alphanumeric-squashed copies)
  - prune_threshold:0.05     (prune near-zero weights below this magnitude)

Usage:
  python scripts/train_base_models.py --models-dir ./models

Source data: schema/eval/ YAML files (fetched by scripts/sources/fetch_*.py).
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # parapet-guardrail/

# Try to locate schema/eval (may be run from project root or scripts/ dir)
CANDIDATE_SCHEMA_DIRS = [
    PROJECT_DIR / "schema" / "eval",
    PROJECT_DIR / "schema",
    Path("schema/eval"),
    Path("schema"),
]
SCHEMA_DIR = next(
    (p for p in CANDIDATE_SCHEMA_DIRS if p.is_dir()),
    PROJECT_DIR / "schema" / "eval",
)

# ---------------------------------------------------------------------------
# Dataset → source file mapping (same as legacy train_l1_specialist.py)
# ---------------------------------------------------------------------------

# Per-category attack source files.
# notinject_benign is included in instruction_override because it contains
# trigger-word FP examples that sharpen the instruction boundary.
ATTACK_SOURCES: dict[str, list[str]] = {
    "instruction_override": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "opensource_jailbreak_cls_attacks.yaml",
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
        "opensource_jailbreak_cls_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "exfiltration": [
        "opensource_chatgpt_jailbreak_attacks.yaml",
        "opensource_jailbreak_cls_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "adversarial_suffix": [
        "opensource_jailbreak_cls_attacks.yaml",
        "opensource_hackaprompt_attacks.yaml",
        "l1_attacks.yaml",
    ],
    "indirect_injection": [
        "opensource_jailbreak_cls_attacks.yaml",
        "opensource_chatgpt_jailbreak_attacks.yaml",
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

# Shared benign sources — hard negatives + general benign.
BENIGN_SOURCES: list[str] = [
    "opensource_no_robots_benign.yaml",
    "opensource_chatgpt_prompts_benign.yaml",
    "opensource_notinject_benign.yaml",       # previously staged/
    "opensource_wildguardmix_benign.yaml",     # previously staged/
    "l1_benign.yaml",
]

# Fetch script map: relative to PROJECT_DIR.
# These scripts write to schema/eval/ (same directory as SCHEMA_DIR).
FETCH_SCRIPT_MAP: dict[str, str] = {
    "opensource_no_robots_benign.yaml":      "scripts/sources/fetch_no_robots.py",
    "opensource_chatgpt_prompts_benign.yaml":"scripts/sources/fetch_chatgpt_prompts.py",
    "opensource_notinject_benign.yaml":      "scripts/sources/fetch_notinject.py",
    "opensource_wildguardmix_benign.yaml":   "scripts/sources/fetch_wildguardmix.py",
    "opensource_chatgpt_jailbreak_attacks.yaml": "scripts/sources/fetch_chatgpt_jailbreak.py",
    "opensource_jailbreak_cls_attacks.yaml":     "scripts/sources/fetch_jailbreak_cls.py",
    "opensource_hackaprompt_attacks.yaml":       "scripts/sources/fetch_hackaprompt.py",
}

# ---------------------------------------------------------------------------
# Hyperparameters — aligned exactly with legacy train_l1_specialist.py
# ---------------------------------------------------------------------------

# All specialists use the same vectorizer config.
# char_wb is required by the Mirror Design Pattern (arxiv 2603.11875v1 §4):
# character n-grams generalise over morphological variations and encoding tricks.
# word n-grams memorise surface tokens and break under trivial rephrasing.
VECTORIZER_CFG = {
    "analyzer":    "char_wb",
    "ngram_range": (3, 5),
    "max_features": 25_000,
    "min_df":       5,
    "binary":       True,
}

SVM_CFG = {
    "C":            0.1,
    "class_weight": "balanced",
    "max_iter":     100_000,
    "dual":         False,
    "penalty":      "l2",       # matches train_l1_specialist.py (L2 retains full ngram vocabulary)
}

TRAIN_CFG = {
    "holdout_pct":      0.20,
    "seed":             42,
    "squash_augment":   True,    # double train data with alphanumeric-squashed copies
    "prune_threshold":  0.001,   # matches legacy train_l1_specialist.py (retains ~14,000+ features)
    "cv_folds":         5,       # 5-fold CV on training set
}

# Legacy train_l1.py hyperparameters (L1 penalty, sparse weights ~300 features).
# Used exclusively for allrounder_legacy to replicate the original baseline.
SVM_CFG_LEGACY = {
    "C":            0.1,
    "class_weight": "balanced",
    "max_iter":     100_000,
    "dual":         False,
    "penalty":      "l1",       # matches train_l1.py baseline (L1 prunes to ~300 sparse weights)
}

TRAIN_CFG_LEGACY = {
    "holdout_pct":      0.20,
    "seed":             42,
    "squash_augment":   True,    # same augment as legacy
    "prune_threshold":  0.001,   # same threshold; L1 already zeroes most features
    "cv_folds":         5,
}

# ---------------------------------------------------------------------------
# Text helpers (port of legacy squash() and L0 transform)
# ---------------------------------------------------------------------------

INVALID_YAML_CTRL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def strip_invalid_yaml_controls(text: str) -> str:
    return INVALID_YAML_CTRL_RE.sub("", text)


def squash(text: str) -> str:
    """Mirror l1.rs::squash() — casefold then keep only alphanumeric (matches Rust behavior)."""
    return "".join(c for c in text.casefold() if c.isalnum())


# ---------------------------------------------------------------------------
# Dataset auto-download
# ---------------------------------------------------------------------------

def ensure_dataset_files(force: bool = False) -> None:
    """Auto-download missing raw dataset files using fetch_*.py scripts.

    Passes HF_TOKEN from environment to each subprocess so gated datasets
    (wildguardmix, etc.) can be downloaded without interactive login.
    The fetch scripts output YAML files directly to schema/eval/.
    """
    import subprocess

    env = os.environ.copy()
    # HF_TOKEN should already be in environment; no need to inject separately
    # since fetch_*.py reads os.environ.get("HF_TOKEN") directly.

    missing = []
    for filename in FETCH_SCRIPT_MAP:
        target = SCHEMA_DIR / filename
        if not target.exists() or force:
            missing.append(filename)

    if not missing:
        print("All dataset files present — skipping download.", file=sys.stderr)
        return

    print(f"\nFetching {len(missing)} missing dataset files...", file=sys.stderr)

    for filename in missing:
        script_path = PROJECT_DIR / FETCH_SCRIPT_MAP[filename]
        if not script_path.exists():
            print(f"  WARN: fetch script not found: {script_path}", file=sys.stderr)
            continue

        print(f"  Fetching {filename} via {script_path.name}...", file=sys.stderr)
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                env=env,
                cwd=str(PROJECT_DIR),  # run from project root so schema/eval/ paths resolve
            )
            # The fetch script for notinject/wildguardmix writes to schema/eval/ (not staging/)
            # Handle the case where the script still writes to staging/
            staging_path = SCHEMA_DIR / "staging" / filename
            canonical_path = SCHEMA_DIR / filename
            if staging_path.exists() and not canonical_path.exists():
                print(f"    Moving {staging_path} → {canonical_path}", file=sys.stderr)
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                staging_path.rename(canonical_path)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: fetch failed for {filename}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# YAML data loading
# ---------------------------------------------------------------------------

def load_yaml_texts(path: Path, label: int) -> list[tuple[str, int]]:
    """Load texts + labels from a YAML file. Handles list-of-str and list-of-dict.

    Expects dicts with 'content' (legacy format) or 'text'/'prompt' keys.
    Skips entries without a matching label field when loading from mixed files.
    """
    if not path.exists():
        return []
    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = strip_invalid_yaml_controls(raw_text)
        data = yaml.load(cleaned, Loader=YAML_LOADER)
    except Exception as e:
        print(f"  WARN: could not load {path}: {e}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []

    target_label_str = "malicious" if label == 1 else "benign"
    results = []
    for item in data:
        if isinstance(item, str):
            text = strip_invalid_yaml_controls(item).strip()
            if text:
                results.append((text, label))
        elif isinstance(item, dict):
            # Support both legacy format (content + label) and simple format (text only)
            item_label = item.get("label", "")
            # If the file has explicit labels, filter to the matching one.
            # If no label field, accept all (e.g. l1_attacks.yaml, l1_benign.yaml).
            if item_label and item_label not in (target_label_str, "positive", "negative", str(label)):
                # Normalize positive/negative to malicious/benign
                if item_label == "positive":
                    item_label = "malicious"
                elif item_label == "negative":
                    item_label = "benign"
                if item_label != target_label_str:
                    continue
            text = (item.get("content") or item.get("text") or
                    item.get("prompt") or "").strip()
            text = strip_invalid_yaml_controls(str(text)).strip()
            if text:
                results.append((text, label))
    return results


def load_attacks_for_category(category: str) -> list[tuple[str, int]]:
    sources = ATTACK_SOURCES.get(category, [])
    records: list[tuple[str, int]] = []
    for src in sources:
        path = SCHEMA_DIR / src
        loaded = load_yaml_texts(path, 1)
        records.extend(loaded)
        if not loaded:
            # Only warn if the file doesn't exist at all — empty is noise
            if not path.exists():
                print(f"  WARN: missing source {path}", file=sys.stderr)
    # Deduplicate
    seen: set[str] = set()
    unique = []
    for text, lbl in records:
        if text not in seen:
            seen.add(text)
            unique.append((text, lbl))
    return unique


def load_benign() -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    for src in BENIGN_SOURCES:
        path = SCHEMA_DIR / src
        loaded = load_yaml_texts(path, 0)
        records.extend(loaded)
        if not loaded and not path.exists():
            print(f"  WARN: missing benign source {path}", file=sys.stderr)
    seen: set[str] = set()
    unique = []
    for text, lbl in records:
        if text not in seen:
            seen.add(text)
            unique.append((text, lbl))
    return unique


# ---------------------------------------------------------------------------
# Training and weight export
# ---------------------------------------------------------------------------

def train_and_emit(
    name: str,
    attacks: list[tuple[str, int]],
    benign: list[tuple[str, int]],
    out_path: Path,
    dry_run: bool = False,
    svm_cfg: dict | None = None,
    train_cfg: dict | None = None,
) -> dict:
    """Train LinearSVC and emit .weights.json. Returns metrics dict."""

    if not attacks:
        print(f"  {name}: SKIPPED — no attack data", file=sys.stderr)
        return {}
    if not benign:
        print(f"  {name}: SKIPPED — no benign data", file=sys.stderr)
        return {}
    _svm   = svm_cfg   if svm_cfg   is not None else SVM_CFG
    _train = train_cfg if train_cfg is not None else TRAIN_CFG

    # Combine all data (no class balancing by truncation — class_weight=balanced handles it)
    all_texts  = [t for t, _ in attacks] + [t for t, _ in benign]
    all_labels = [1] * len(attacks) + [0] * len(benign)

    print(f"\n  {name}: {len(attacks)} attacks + {len(benign)} benign", file=sys.stderr)

    if dry_run:
        print(f"  [dry-run] skipping fit", file=sys.stderr)
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "samples": len(all_texts)}

    # Squash augmentation: double training data with alphanumeric-squashed copies.
    # Mirrors legacy --squash-augment flag (train_l1_specialist.py §4b).
    if _train["squash_augment"]:
        aug_texts  = all_texts + [squash(t) for t in all_texts]
        aug_labels = all_labels + all_labels
    else:
        aug_texts, aug_labels = all_texts, all_labels

    # Stratified 80/20 split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            aug_texts, aug_labels,
            test_size=_train["holdout_pct"],
            stratify=aug_labels,
            random_state=_train["seed"],
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            aug_texts, aug_labels,
            test_size=_train["holdout_pct"],
            random_state=_train["seed"],
        )

    vec = CountVectorizer(
        analyzer=VECTORIZER_CFG["analyzer"],
        ngram_range=VECTORIZER_CFG["ngram_range"],
        max_features=VECTORIZER_CFG["max_features"],
        min_df=VECTORIZER_CFG["min_df"],
        binary=VECTORIZER_CFG["binary"],
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    # 5-fold CV on training set (mirrors legacy)
    n_folds = _train["cv_folds"]
    if n_folds >= 2 and X_tr.shape[0] >= n_folds * 2:
        clf_cv = LinearSVC(**_svm)
        cv_scores = cross_val_score(clf_cv, X_tr, y_train, cv=n_folds, scoring="f1")
        print(f"  {name}: {n_folds}-fold CV F1={cv_scores.mean():.4f} (±{cv_scores.std():.4f})",
              file=sys.stderr)

    clf = LinearSVC(**_svm)
    clf.fit(X_tr, y_train)

    preds = clf.predict(X_te)
    f1  = float(f1_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    pre = float(precision_score(y_test, preds, zero_division=0))
    print(f"  {name}: holdout F1={f1:.3f}  Recall={rec:.3f}  Precision={pre:.3f}  "
          f"train_samples={len(X_train)}  features={X_tr.shape[1]}", file=sys.stderr)

    # Extract weights (prune near-zero)
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    prune_thr = _train["prune_threshold"]
    weights = {
        str(nm): float(w)
        for nm, w in zip(feature_names, coef)
        if abs(w) >= prune_thr
    }
    bias = float(clf.intercept_[0])

    print(f"  {name}: {len(weights)} non-pruned weights (|w|>={prune_thr})", file=sys.stderr)

    # Top attack / benign indicators
    sorted_w = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  Top attack n-grams: {[k for k, _ in sorted_w[:5]]}", file=sys.stderr)
    print(f"  Top benign n-grams: {[k for k, _ in sorted_w[-5:]]}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "bias": bias,
            "weights": weights,
            "analyzer": VECTORIZER_CFG["analyzer"],
            "ngram_range": list(VECTORIZER_CFG["ngram_range"]),
        }, f, ensure_ascii=False)

    return {"f1": f1, "recall": rec, "precision": pre,
            "samples": len(all_texts), "weights": len(weights)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train all 9 base SVM models")
    parser.add_argument("--models-dir", default="./models",
                        help="Directory to write base/*.weights.json files")
    parser.add_argument("--schema-dir", default=None,
                        help="Override schema/eval directory path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs without training")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if .weights.json already exists")
    parser.add_argument("--force-fetch", action="store_true",
                        help="Re-download dataset files even if present")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Do not run fetch scripts even if files are missing")
    args = parser.parse_args()

    global SCHEMA_DIR
    if args.schema_dir:
        SCHEMA_DIR = Path(args.schema_dir)

    models_dir = Path(args.models_dir) / "base"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Schema dir: {SCHEMA_DIR}", file=sys.stderr)
    print(f"Models dir: {models_dir}", file=sys.stderr)

    # Step 1: Auto-download missing datasets.
    if not args.skip_fetch:
        ensure_dataset_files(force=args.force_fetch)
    else:
        print("Skipping dataset fetch (--skip-fetch).", file=sys.stderr)

    # Step 2: Load benign corpus (shared across all specialists).
    print("\nLoading benign corpus...", file=sys.stderr)
    benign = load_benign()
    print(f"Benign total: {len(benign)} records", file=sys.stderr)
    if not benign:
        print("ERROR: No benign data found. Run fetch scripts first.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Load all attack sources across categories.
    all_attacks: list[tuple[str, int]] = []
    metrics_summary: dict[str, dict] = {}
    for cat in ATTACK_SOURCES:
        atks = load_attacks_for_category(cat)
        all_attacks.extend(atks)

    print(f"\nTraining allrounder base SVM on combined attack records...", file=sys.stderr)
    seen: set[str] = set()
    unique_attacks: list[tuple[str, int]] = []
    for text, lbl in all_attacks:
        if text not in seen:
            seen.add(text)
            unique_attacks.append((text, lbl))

    out_path = models_dir / "allrounder.weights.json"
    if not out_path.exists() or args.force:
        m = train_and_emit("allrounder", unique_attacks, benign, out_path, dry_run=args.dry_run)
        if m:
            metrics_summary["allrounder"] = m
    else:
        print(f"  allrounder: cache hit — skipping (use --force to retrain)", file=sys.stderr)

    # Step 4: Train allrounder_legacy — identical dataset, but legacy train_l1.py L1 config
    #   penalty="l1", sparse weights (~300), matching the original baseline allrounder
    legacy_path = models_dir / "allrounder_legacy.weights.json"
    if not legacy_path.exists() or args.force:
        print(f"\nTraining allrounder_legacy (L1 penalty — legacy train_l1.py config)...", file=sys.stderr)
        m = train_and_emit(
            "allrounder_legacy", unique_attacks, benign, legacy_path,
            dry_run=args.dry_run,
            svm_cfg=SVM_CFG_LEGACY,
            train_cfg=TRAIN_CFG_LEGACY,
        )
        if m:
            metrics_summary["allrounder_legacy"] = m
    else:
        print(f"  allrounder_legacy: cache hit — skipping (use --force to retrain)", file=sys.stderr)

    # Step 5: Summary
    print("\n=== Base Model Training Summary ===", file=sys.stderr)
    print(f"{'Model':<25} {'F1':>6} {'Recall':>8} {'Precision':>10} {'Samples':>8} {'Weights':>8}",
          file=sys.stderr)
    print("-" * 68, file=sys.stderr)
    for name, m in metrics_summary.items():
        print(f"  {name:<23} {m.get('f1',0):>6.3f} {m.get('recall',0):>8.3f} "
              f"{m.get('precision',0):>10.3f} {m.get('samples',0):>8} {m.get('weights',0):>8}",
              file=sys.stderr)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
