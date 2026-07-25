# SVM Specialist Training — Legacy Parity Implementation Plan

## Goal

Replace the current `train_base_models.py` in `parapet-guardrail/` with an
approach that **exactly replicates** how the 9 models (`l1_weights.rs` +
`l1_weights_*.rs`) were built in the legacy `parapet/` codebase — same training
script, same hyperparameters, same datasets, same preprocessing, zero deviation.

The `.weights.json` runtime format is unchanged. Only the training pipeline changes.

---

## Research Summary

### What the Legacy Models Actually Used (from `l1_weights_*.rs` frontmatter)

The existing `train_base_models.py` in parapet-guardrail is **wrong in multiple ways**.
It uses the wrong data sources AND wrong hyperparameters for every specialist.

| Specialist | Analyzer | N-gram | Key datasets (from frontmatter) |
|---|---|---|---|
| `generalist` (allrounder) | `char_wb` | 3–5 | 76 files from mirror v4 curated workflow |
| `instruction_override` | `word` | 1–3 | `global_benign_train.yaml`, `l1_attacks.yaml`, `opensource_gandalf_attacks.yaml`, `opensource_tensortrust_hijacking_attacks.yaml` |
| `roleplay_jailbreak` | `word` | 2–4 | `global_benign_train.yaml`, `opensource_chatgpt_jailbreak_attacks.yaml`, `opensource_giskard_attacks.yaml`, `opensource_imoxto_attacks.yaml`, `opensource_jailbreak_cls_attacks.yaml`, `opensource_jailbreakv_attacks.yaml` |
| `meta_probe` | `word` | 1–2 | `global_benign_train.yaml`, `l1_attacks.yaml`, `opensource_gandalf_attacks.yaml`, `opensource_giskard_attacks.yaml`, `opensource_tensortrust_extraction_attacks.yaml`, `thewall_elite-attack_pos.yaml` |
| `exfiltration` | `char_wb` | 3–5 | `global_benign_train.yaml`, `l1_attacks.yaml`, `opensource_bipia_attacks.yaml`, `opensource_imoxto_attacks.yaml` |
| `adversarial_suffix` | `char` | 3–5 | `thewall_amplegcg_pos.yaml`, `thewall_jailbreakbench_pos.yaml`, `thewall_llm-attacks_pos.yaml` |
| `indirect_injection` | `char_wb` | 3–5 | `global_benign_train.yaml`, `opensource_bipia_attacks.yaml`, `opensource_llmail_attacks.yaml`, `thewall_agentic-rag-redteam-bench_pos.yaml`, `thewall_atlas_pos.yaml` |
| `obfuscation` | `char_wb` | 3–5 | `obfuscation_curated_attacks.yaml` + 6 benign sources |
| `constraint_bypass` | `char_wb` | 3–5 | 29 files — many `thewall_*` attack/benign sources |

### Hyperparameter Differences vs. Current `train_base_models.py`

| Parameter | Current (wrong) | Legacy (correct) |
|---|---|---|
| `C` | `1.0` | `0.1` |
| `penalty` | `'l2'` (default) | `'l1'` |
| `dual` | `True` (default) | `False` (required for L1) |
| `max_iter` | `2000` | `100000` |
| `tol` | `1e-3` (default) | `1e-4` |
| `class_weight` | not set | `'balanced'` |
| `min_df` | `2` | `5` |
| `holdout split` | 15% random | 20% stratified, seed=42 |
| `squash augment` | always applied | NOT applied |
| `l0 transform` | not applied | `--apply-l0-transform` |
| `prune threshold` | `w != 0.0` | `abs(w) >= 0.05` |

### Dataset Availability in `parapet-guardrail/schema/eval/`

**Already present:**
- `opensource_chatgpt_jailbreak_attacks.yaml` ✅
- `opensource_gandalf_attacks.yaml` ✅
- `opensource_giskard_attacks.yaml` ✅
- `opensource_hackaprompt_attacks.yaml` ✅
- `opensource_jailbreak_cls_attacks.yaml` ✅
- `opensource_notinject_benign.yaml` ✅
- `opensource_wildguardmix_benign.yaml` ✅
- `opensource_no_robots_benign.yaml` ✅
- `l1_attacks.yaml` ✅
- `l1_benign.yaml` ✅

**Missing — need to be fetched or copied:**

| File | Source | Notes |
|---|---|---|
| `global_benign_train.yaml` | Private — legacy `schema/eval/` | Primary benign set for 5 specialists (19,389 records) |
| `opensource_bipia_attacks.yaml` | HuggingFace → `fetch_bipia.py` | exfiltration + indirect_injection |
| `opensource_imoxto_attacks.yaml` | HuggingFace → `fetch_imoxto.py` | roleplay_jailbreak + exfiltration |
| `opensource_jailbreakv_attacks.yaml` | HuggingFace → `fetch_jailbreakv.py` | roleplay_jailbreak |
| `opensource_llmail_attacks.yaml` | HuggingFace → `fetch_llmail.py` | indirect_injection |
| `opensource_tensortrust_hijacking_attacks.yaml` | HuggingFace → `fetch_tensor_trust.py` | instruction_override |
| `opensource_tensortrust_extraction_attacks.yaml` | HuggingFace → `fetch_tensor_trust.py` | meta_probe |
| `obfuscation_curated_attacks.yaml` | Private curation only | obfuscation (sole attack source) |
| `opensource_alpaca_benign.yaml` | HuggingFace → `fetch_alpaca.py` | obfuscation + constraint_bypass |
| `opensource_hc3_benign.yaml` | HuggingFace → `fetch_hc3.py` | obfuscation |
| `opensource_protectai_val_benign.yaml` | HuggingFace → `fetch_protectai_validation.py` | obfuscation |
| `opensource_wildchat_benign.yaml` | HuggingFace → `fetch_wildchat.py` | obfuscation + constraint_bypass |
| `opensource_ultrachat_benign.yaml` | HuggingFace → `fetch_ultrachat.py` | constraint_bypass |
| `opensource_jbb_paraphrase_attacks.yaml` | HuggingFace → `fetch_jbb_paraphrase.py` | constraint_bypass |
| `thewall_*` files (15+ files) | **Private TheWall corpus** | constraint_bypass + adversarial_suffix + meta_probe + indirect_injection |

> [!CAUTION]
> `thewall_*`, `global_benign_train.yaml`, and `obfuscation_curated_attacks.yaml`
> are **local-only private files** — they cannot be downloaded from any public source.
> They must be copied from the legacy `parapet/schema/eval/` location if present there,
> or accessed via a configured `LEGACY_SCHEMA_DIR` path.

---

## Open Questions

> [!IMPORTANT]
> **Where are `thewall_*` files, `global_benign_train.yaml`, and
> `obfuscation_curated_attacks.yaml` stored locally?**
> The legacy `schema/eval/` directory only shows 30 files — these private files are
> not there. Are they in a separate data volume, parapet-data folder, or external location?
> Locating these is the single highest-priority step to maximize parity.

> [!IMPORTANT]
> **Allrounder model**: The `l1_weights.rs` generalist was trained via the mirror v4
> curated workflow (76 files, many private). For the guardrail allrounder, should we:
> (a) Use all open-source attack data from the 8 specialists combined (best achievable without private data), or
> (b) Wait until private data is located before training it?

---

## Proposed Changes

### Architecture

```
cargo run
   │
   ▼  (Rust startup check — unchanged)
Are all 9 .weights.json present?
   │ No
   ▼
scripts/setup_training.py      ← NEW orchestrator (replaces train_base_models.py)
   │
   ├─ 1. ensure_open_source_datasets()   fetch missing HuggingFace datasets
   ├─ 2. locate_private_datasets()       copy from LEGACY_SCHEMA_DIR if present
   ├─ 3. train_specialist() × 9         invoke train_l1_specialist.py per model
   └─ 4. emit .weights.json             via new --out-weights-json flag
   │
   ▼
Server starts
```

---

### Component 1 — Copy `train_l1_specialist.py` Verbatim (+ One New Flag)

**Action**: Copy `scripts/train_l1_specialist.py` from the legacy `parapet/`
project into `parapet-guardrail/scripts/train_l1_specialist.py` **with zero
changes to training logic**.

**Only addition** — one new output flag that emits `.weights.json` instead of `.rs`:

```python
# In argparse (new non-breaking arg):
parser.add_argument("--out-weights-json", type=str, default=None,
    help="If set, emit a .weights.json alongside (or instead of) the .rs file")

# New function (added after codegen_phf call):
def emit_weights_json(bias, weights, out_path, analyzer, ngram_range):
    """Emit .weights.json compatible with parapet-guardrail svm_base.rs."""
    import json
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "bias": bias,
            "weights": weights,
            "analyzer": analyzer,
            "ngram_range": list(ngram_range),
        }, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(weights)} entries to {out_path}", file=sys.stderr)
```

No other changes. `--out` still generates `.rs` files if provided; `--out-weights-json`
generates the `.weights.json`. Both can be provided simultaneously.

---

### Component 2 — Copy Fetch Scripts from Legacy

Copy from `parapet/scripts/sources/` → `parapet-guardrail/scripts/sources/`:

| Script | Dataset |
|---|---|
| `fetch_bipia.py` | `opensource_bipia_attacks.yaml` |
| `fetch_imoxto.py` | `opensource_imoxto_attacks.yaml` |
| `fetch_jailbreakv.py` | `opensource_jailbreakv_attacks.yaml` |
| `fetch_llmail.py` | `opensource_llmail_attacks.yaml` |
| `fetch_tensor_trust.py` | tensortrust hijacking + extraction |
| `fetch_alpaca.py` | `opensource_alpaca_benign.yaml` |
| `fetch_hc3.py` | `opensource_hc3_benign.yaml` |
| `fetch_wildchat.py` | `opensource_wildchat_benign.yaml` |
| `fetch_ultrachat.py` | `opensource_ultrachat_benign.yaml` |
| `fetch_protectai_validation.py` | `opensource_protectai_val_benign.yaml` |
| `fetch_jbb_paraphrase.py` | `opensource_jbb_paraphrase_attacks.yaml` |

All of these exist in the legacy `scripts/sources/` directory already.

---

### Component 3 — `scripts/setup_training.py` (New Orchestrator)

**Replaces** `train_base_models.py`. Responsibilities:

1. **`ensure_open_source_datasets(schema_dir)`** — for each HuggingFace-sourced
   file needed by any specialist, check if it exists; if not, run the fetch script.

2. **`locate_private_datasets(schema_dir, legacy_dir)`** — check `LEGACY_SCHEMA_DIR`
   for private files (`global_benign_train.yaml`, `thewall_*`,
   `obfuscation_curated_attacks.yaml`). Copy them to `schema_dir` if found.
   Log a warning (not an error) for each missing private file and apply the
   fallback substitution.

3. **`build_specialist_manifest(schema_dir)`** — build per-specialist file lists
   (using files that are actually present), incorporating fallbacks for missing
   private data.

4. **`train_all(models_dir, schema_dir)`** — for each of the 9 models:
   - Skip if `.weights.json` already exists (cache-hit)
   - Call `train_l1_specialist.py` with the exact flags per specialist
   - Pass `--out-weights-json` to emit `.weights.json`

5. **Print a parity summary** at the end listing which private datasets were
   found vs. substituted.

#### Exact `train_l1_specialist.py` Invocations Per Specialist

```bash
# instruction_override (word 1-3, sources from frontmatter)
python scripts/train_l1_specialist.py \
  --specialist instruction_override \
  --analyzer word --ngram-min 1 --ngram-max 3 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/l1_attacks.yaml \
      schema/eval/opensource_gandalf_attacks.yaml \
      schema/eval/opensource_tensortrust_hijacking_attacks.yaml \
  --benign-files \
      schema/eval/global_benign_train.yaml \   # or fallback benign set
  --out-weights-json models/base/instruction_override.weights.json

# roleplay_jailbreak (word 2-4)
python scripts/train_l1_specialist.py \
  --specialist roleplay_jailbreak \
  --analyzer word --ngram-min 2 --ngram-max 4 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/opensource_chatgpt_jailbreak_attacks.yaml \
      schema/eval/opensource_giskard_attacks.yaml \
      schema/eval/opensource_imoxto_attacks.yaml \
      schema/eval/opensource_jailbreak_cls_attacks.yaml \
      schema/eval/opensource_jailbreakv_attacks.yaml \
  --benign-files schema/eval/global_benign_train.yaml \
  --out-weights-json models/base/roleplay_jailbreak.weights.json

# meta_probe (word 1-2)
python scripts/train_l1_specialist.py \
  --specialist meta_probe \
  --analyzer word --ngram-min 1 --ngram-max 2 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/l1_attacks.yaml \
      schema/eval/opensource_gandalf_attacks.yaml \
      schema/eval/opensource_giskard_attacks.yaml \
      schema/eval/opensource_tensortrust_extraction_attacks.yaml \
      schema/eval/thewall_elite-attack_pos.yaml \   # private; skip if missing
  --benign-files schema/eval/global_benign_train.yaml \
  --out-weights-json models/base/meta_probe.weights.json

# exfiltration (char_wb 3-5)
python scripts/train_l1_specialist.py \
  --specialist exfiltration \
  --analyzer char_wb --ngram-min 3 --ngram-max 5 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/l1_attacks.yaml \
      schema/eval/opensource_bipia_attacks.yaml \
      schema/eval/opensource_imoxto_attacks.yaml \
  --benign-files schema/eval/global_benign_train.yaml \
  --out-weights-json models/base/exfiltration.weights.json

# adversarial_suffix (char 3-5)
python scripts/train_l1_specialist.py \
  --specialist adversarial_suffix \
  --analyzer char --ngram-min 3 --ngram-max 5 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/thewall_amplegcg_pos.yaml \       # private
      schema/eval/thewall_jailbreakbench_pos.yaml \ # private
      schema/eval/thewall_llm-attacks_pos.yaml \    # private
  --benign-files schema/eval/global_benign_train.yaml \
  --out-weights-json models/base/adversarial_suffix.weights.json

# indirect_injection (char_wb 3-5)
python scripts/train_l1_specialist.py \
  --specialist indirect_injection \
  --analyzer char_wb --ngram-min 3 --ngram-max 5 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/opensource_bipia_attacks.yaml \
      schema/eval/opensource_llmail_attacks.yaml \
      schema/eval/thewall_agentic-rag-redteam-bench_pos.yaml \ # private
      schema/eval/thewall_atlas_pos.yaml \                     # private
  --benign-files schema/eval/global_benign_train.yaml \
  --out-weights-json models/base/indirect_injection.weights.json

# obfuscation (char_wb 3-5)
python scripts/train_l1_specialist.py \
  --specialist obfuscation \
  --analyzer char_wb --ngram-min 3 --ngram-max 5 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/obfuscation_curated_attacks.yaml \ # private — sole attack source
  --benign-files \
      schema/eval/opensource_alpaca_benign.yaml \
      schema/eval/opensource_hc3_benign.yaml \
      schema/eval/opensource_notinject_benign.yaml \
      schema/eval/opensource_protectai_val_benign.yaml \
      schema/eval/opensource_wildchat_benign.yaml \
      schema/eval/opensource_wildguardmix_benign.yaml \
  --out-weights-json models/base/obfuscation.weights.json

# constraint_bypass (char_wb 3-5) — 29 sources
python scripts/train_l1_specialist.py \
  --specialist constraint_bypass \
  --analyzer char_wb --ngram-min 3 --ngram-max 5 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files \
      schema/eval/opensource_hackaprompt_attacks.yaml \
      schema/eval/opensource_jbb_attacks.yaml \
      schema/eval/opensource_jbb_paraphrase_attacks.yaml \
      schema/eval/thewall_ai-safety-50k_pos.yaml \
      schema/eval/thewall_arabic-hallucination-red-teaming_pos.yaml \
      schema/eval/thewall_fractured-sorry-bench-automated-multishot-jailbreak_pos.yaml \
      schema/eval/thewall_galtea-red-teaming-clustered-data_pos.yaml \
      schema/eval/thewall_generative-ai-red-teaming_pos.yaml \
      schema/eval/thewall_harmbench_pos.yaml \
      schema/eval/thewall_harmful-tasks_pos.yaml \
      schema/eval/thewall_jailbreakbench_pos.yaml \
      schema/eval/thewall_multijail_pos.yaml \
      schema/eval/thewall_resa_pos.yaml \
      schema/eval/thewall_wildjailbreak_pos.yaml \
  --benign-files \
      schema/eval/opensource_alpaca_benign.yaml \
      schema/eval/opensource_chatgpt_prompts_benign.yaml \
      schema/eval/opensource_no_robots_benign.yaml \
      schema/eval/opensource_notinject_benign.yaml \
      schema/eval/opensource_ultrachat_benign.yaml \
      schema/eval/opensource_wildchat_benign.yaml \
      schema/eval/opensource_wildguardmix_benign.yaml \
      schema/eval/thewall_cohereforai-aya-dataset-arabic_neg.yaml \
      schema/eval/thewall_databricks-dolly-15k-chinese_neg.yaml \
      schema/eval/thewall_databricks-dolly-15k_neg.yaml \
      schema/eval/thewall_ru-turbo-alpaca_neg.yaml \
      schema/eval/thewall_trivia-qa_neg.yaml \
      schema/eval/thewall_wildchat-1m_neg.yaml \
      schema/eval/thewall_writingprompts_neg.yaml \
      schema/eval/thewall_xstest_neg.yaml \
  --out-weights-json models/base/constraint_bypass.weights.json

# allrounder (char_wb 3-5) — all attack files from all 8 categories combined
python scripts/train_l1_specialist.py \
  --specialist allrounder \
  --analyzer char_wb --ngram-min 3 --ngram-max 5 \
  --c 0.1 --min-df 5 --holdout-pct 0.2 --seed 42 \
  --apply-l0-transform \
  --attack-files [all attack files from all 8 specialists above] \
  --benign-files schema/eval/global_benign_train.yaml \
  --out-weights-json models/base/allrounder.weights.json
```

#### Fallback Strategy for Missing Private Datasets

| Private file | Used by | Fallback if missing |
|---|---|---|
| `global_benign_train.yaml` | 5 specialists (benign) | `opensource_no_robots_benign.yaml` + `opensource_wildguardmix_benign.yaml` + `l1_benign.yaml` |
| `thewall_elite-attack_pos.yaml` | meta_probe | Skip (reduces attack count slightly) |
| `thewall_agentic-rag-redteam-bench_pos.yaml` | indirect_injection | Use bipia + llmail only |
| `thewall_atlas_pos.yaml` | indirect_injection | Skip (176 records) |
| `thewall_amplegcg_pos.yaml` | adversarial_suffix | `opensource_jailbreak_cls_attacks.yaml` |
| `thewall_llm-attacks_pos.yaml` | adversarial_suffix | Skip |
| `thewall_jailbreakbench_pos.yaml` | adversarial_suffix + constraint_bypass | `opensource_jbb_attacks.yaml` |
| `thewall_*` other (constraint_bypass) | constraint_bypass | `opensource_hackaprompt_attacks.yaml` + `opensource_jbb_attacks.yaml` |
| `obfuscation_curated_attacks.yaml` | obfuscation **only** | `opensource_deepset_attacks.yaml` — **different model; operator warned explicitly** |

---

### Component 4 — `.env` / `.env.example` Additions

```dotenv
# ── Training Data ──────────────────────────────────────────────────────────
# Path to legacy parapet schema/eval directory (for private datasets).
# Auto-detects ../parapet/schema/eval if not set.
LEGACY_SCHEMA_DIR=../parapet/schema/eval

# Directory for all training YAML files (public + private combined).
TRAINING_DATA_DIR=./schema/eval
```

---

### Component 5 — `src/main.rs` (One Line Change)

```rust
// BEFORE:
.arg("scripts/train_base_models.py")

// AFTER:
.arg("scripts/setup_training.py")
```

The `--models-dir` flag and all surrounding logic remain identical.

---

### File Change Summary

```
parapet-guardrail/
  scripts/
    setup_training.py          [NEW]    replaces train_base_models.py
    train_l1_specialist.py     [NEW]    copied from legacy + --out-weights-json flag
    train_base_models.py       [DELETE] incorrect; replaced
    sources/
      fetch_bipia.py           [COPY from legacy]
      fetch_imoxto.py          [COPY]
      fetch_jailbreakv.py      [COPY]
      fetch_llmail.py          [COPY]
      fetch_tensor_trust.py    [COPY]
      fetch_alpaca.py          [COPY]
      fetch_hc3.py             [COPY]
      fetch_wildchat.py        [COPY]
      fetch_ultrachat.py       [COPY]
      fetch_protectai_validation.py  [COPY]
      fetch_jbb_paraphrase.py  [COPY]
  src/
    main.rs                    [MODIFY] train_base_models → setup_training (1 line)
  .env / .env.example          [MODIFY] add LEGACY_SCHEMA_DIR, TRAINING_DATA_DIR
```

---

## Verification Plan

```bash
cd parapet-guardrail

# Dry-run: verify dataset resolution without training
python scripts/setup_training.py --models-dir ./models --dry-run

# Train one specialist end-to-end
python scripts/setup_training.py --models-dir ./models --only instruction_override

# Full run
python scripts/setup_training.py --models-dir ./models

# Start server (verifies .weights.json loading)
cargo run
```

**Manual checks**:
- Feature count in `instruction_override.weights.json` should be ~203
- Feature count in `adversarial_suffix.weights.json` should be ~492
- `cargo run` skips training on second run (files present)
- `POST /v1/detect` with injection text returns score > 0

---

## Parity Risk Summary

| Specialist | Private data needed | Achievable parity |
|---|---|---|
| instruction_override | `global_benign_train.yaml` | ~90% with fallback benign |
| roleplay_jailbreak | `global_benign_train.yaml` | ~90% |
| meta_probe | `global_benign_train.yaml` + 1 thewall file | ~85% |
| exfiltration | `global_benign_train.yaml` | ~90% |
| adversarial_suffix | 3 thewall attack files | ~60% (major sources private) |
| indirect_injection | `global_benign_train.yaml` + 2 thewall files | ~75% |
| obfuscation | `obfuscation_curated_attacks.yaml` (only source!) | ~40% — fundamentally different model |
| constraint_bypass | 8 thewall attack + 4 thewall benign files | ~55% |
| allrounder | All of the above | ~75% combined |

> [!CAUTION]
> `obfuscation` is the highest-risk specialist. Its sole attack source is
> a private curation. Any public substitute will produce a materially different
> model. The operator will be warned explicitly at training time.

> [!TIP]
> If `LEGACY_SCHEMA_DIR` points to a location where the private files exist,
> near-full parity is achievable for instruction_override, roleplay_jailbreak,
> meta_probe, exfiltration, and indirect_injection. Locating that path is the
> single most impactful action before running the pipeline.
