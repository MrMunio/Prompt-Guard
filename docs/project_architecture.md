# Parapet Guardrail Engine — Project Architecture

> **Last Updated:** 2026-07-27
> This document describes the current as-built architecture after all implementation phases are complete.

---

## Overview

`parapet-guardrail` is a **self-contained, operator-deployable prompt-injection guardrail API** built in Rust (axum). It provides:

- Per-request text scoring via configurable SVM classifiers and regex scanners
- 9 pre-compiled, PHF-accelerated base SVM models (auto-built on first startup)
- Client-registered custom SVM models trained via REST API (async, background)
- Client-registered custom regex pattern groups (LLM-assisted, opt-in)
- L0 normalization on every request (NFKC, HTML strip, zero-width removal)
- Single-key API authentication (constant-time comparison)
- SQLite (dev) and Postgres (prod) with auto-migration
- Dataset catalog for blending open-source training data into custom models

---

## Directory Structure

```
parapet-guardrail/
├── README.md
├── .env / .env.example                  ← Environment configuration
├── Cargo.toml                           ← Independent workspace (standalone)
├── Dockerfile                           ← Multi-stage: Rust + Python runtime
├── docker-compose.yml                   ← Dev/prod compose with volumes
├── parapet.yaml                         ← Local parapet L3 pattern config
│
├── schema/                              ← Dataset YAML files for base models
├── models/
│   ├── base/                            ← 9 PHF-compiled base weight files
│   │   ├── allrounder.weights.json
│   │   ├── instruction_override.weights.json
│   │   ├── roleplay_jailbreak.weights.json
│   │   ├── meta_probe.weights.json
│   │   ├── exfiltration.weights.json
│   │   ├── adversarial_suffix.weights.json
│   │   ├── indirect_injection.weights.json
│   │   ├── obfuscation.weights.json
│   │   └── constraint_bypass.weights.json
│   ├── custom/                          ← Client-trained model weight files (gitignored)
│   │   └── {uuid}.weights.json
│   └── base_cache/                      ← Per-category JSONL blend caches (training speedup)
│       ├── _benign.jsonl
│       └── {category}.jsonl
│
├── src/
│   ├── main.rs                          ← Startup orchestration + axum server
│   ├── config.rs                        ← AppConfig from env vars
│   ├── auth.rs                          ← X-API-Key middleware
│   ├── error.rs                         ← ApiError → HTTP responses
│   ├── datasets.rs                      ← Startup dataset catalog seeder
│   ├── db/
│   │   ├── mod.rs                       ← DbPool enum: Sqlite | Postgres
│   │   └── migrations.rs                ← Embedded SQL schema; runs on connect
│   ├── engine/
│   │   ├── mod.rs                       ← GuardrailEngine + EngineState (shared state)
│   │   ├── l0.rs                        ← L0 normalization wrapper
│   │   ├── scoring.rs                   ← N-gram tokenization + SVM dot-product
│   │   ├── svm_base.rs                  ← BaseModelRegistry: 9 weight files at startup
│   │   ├── svm_custom.rs                ← CustomModelCache: lazy-load + LRU eviction
│   │   ├── regex_base.rs                ← Built-in L3 scanner (parapet DefaultInboundScanner)
│   │   ├── regex_custom.rs              ← Custom pattern group cache + regex scan
│   │   └── verdict.rs                   ← DetectResponse + GuardrailResult types
│   └── api/
│       ├── mod.rs                       ← Router: Axum 0.8 {id} path syntax + auth middleware
│       ├── health.rs                    ← GET /v1/health
│       ├── detect.rs                    ← POST /v1/detect
│       ├── patterns.rs                  ← CRUD /v1/patterns
│       ├── models.rs                    ← CRUD /v1/models
│       ├── datasets.rs                  ← GET /v1/datasets + POST /v1/datasets/{id}/fetch
│       └── train.rs                     ← POST /v1/models/{id}/train + training-status
│
└── scripts/
    ├── train_base_models.py             ← Train all 9 base SVMs from schema/eval/ YAML files
    ├── train_custom_model.py            ← Train one custom SVM; emit weights + metrics JSON
    ├── mirror_augment.py                ← LLM mirror augmentation (Mirror Design Pattern §4.2)
    ├── generate_regex.py                ← LLM regex generation from plain-text descriptions
    └── sources/                         ← Dataset fetch scripts (fetch_no_robots.py, etc.)
```

---

## Request Lifecycle

```
Client HTTP Request
       │
       ▼
┌─────────────────┐
│   auth.rs       │  X-API-Key header check (constant-time)
│   middleware    │  → 401 if missing or wrong
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   api/detect.rs │  Parse guardrail selector fields
│                 │  Validate text length (MAX_TEXT_CHARS)
│                 │  Resolve custom model/pattern IDs → 404 if unknown
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   engine/l0.rs  │  L0 normalization (always):
│                 │  HTML strip → zero-width removal → NFKC → control char cleanup
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Scoring                             │
│                                                                 │
│  svm_base.rs       svm_custom.rs       regex_base.rs           │
│  PHF weight maps   LRU-cached          parapet L3 scanner       │
│  (zero I/O)        .weights.json       category-filtered        │
│                                                                 │
│                    regex_custom.rs                              │
│                    DB-stored regex     compiled on demand        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   verdict.rs    │  Aggregate per-guardrail results
                    │                 │  Compute composite_score + verdict
                    └────────┬────────┘
                             │
                             ▼
                    JSON Response  →  Client
```

---

## Startup Sequence

`src/main.rs` runs these steps in order at boot:

1. Load `.env` from current directory → populate `AppConfig`
2. Connect to DB (`DATABASE_URL`) → run embedded SQL migrations (create tables if not exist)
3. **Base model cache check**: scan `models/base/` for all 9 `.weights.json` files. If any are missing → invoke `scripts/train_base_models.py` (blocking, may take several minutes on first run)
4. Load all 9 base model weight files into memory in `BaseModelRegistry`
5. Seed dataset catalog from `schema/eval/` YAML files into DB (non-blocking background task)
6. Start axum server on `0.0.0.0:{PORT}` (default 9900)

---

## Engine State (`src/engine/mod.rs`)

`EngineState` is an `Arc`-wrapped struct shared across all request handlers:

```rust
pub struct EngineState {
    pub engine: GuardrailEngine,
    pub max_text_chars: usize,
    pub llm_base_url: String,
    pub llm_model: String,
    pub llm_api_key: String,
    pub python_executable: String,
    /// Max mirror records generated per label class per training request.
    /// Controls LLM spend when enable_mirror=true.
    pub mirror_max_records: usize,
}
```

`GuardrailEngine` holds:
- `base_models: Arc<BaseModelRegistry>` — 9 pre-loaded SVMs
- `custom_models: Arc<CustomModelCache>` — LRU-cached custom SVMs
- `regex_base: Arc<BaseRegexScanner>` — compiled L3 pattern scanner
- `regex_custom_cache: Arc<CustomRegexCache>` — compiled custom regex groups
- `models_dir: String` — path to `./models`

---

## Database Schema

```sql
-- Custom regex pattern groups
CREATE TABLE pattern_groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    category    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Individual compiled patterns in a group
CREATE TABLE pattern_entries (
    id         TEXT PRIMARY KEY,
    group_id   TEXT NOT NULL REFERENCES pattern_groups(id) ON DELETE CASCADE,
    raw_input  TEXT NOT NULL,  -- original user input (text or regex)
    pattern    TEXT NOT NULL,  -- compiled regex stored (may be LLM-generated)
    source     TEXT NOT NULL,  -- "user_regex" | "llm_generated"
    created_at TEXT NOT NULL
);

-- Custom SVM model registry
CREATE TABLE custom_models (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    description      TEXT,
    category         TEXT NOT NULL,
    status           TEXT NOT NULL,  -- "pending"|"training"|"ready"|"error"
    model_path       TEXT,
    training_samples INTEGER,        -- TOTAL records model trained on (client + blends)
    f1_score         REAL,
    error_message    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- Per-model training records (client submitted + mirror generated)
CREATE TABLE training_records (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL REFERENCES custom_models(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    label         INTEGER NOT NULL,  -- 0 = benign, 1 = attack
    source        TEXT NOT NULL,     -- "client" | "mirror_generated"
    mirror_of     TEXT,
    base_category TEXT,
    created_at    TEXT NOT NULL
);

-- Open-source dataset catalog
CREATE TABLE training_datasets (
    id               TEXT PRIMARY KEY,
    file_name        TEXT NOT NULL,
    display_name     TEXT NOT NULL,
    description      TEXT,
    category         TEXT,
    label_type       TEXT,           -- "attack_only" | "benign_only" | "mixed"
    record_count     INTEGER,
    attack_count     INTEGER,
    benign_count     INTEGER,
    file_path        TEXT,
    fetch_status     TEXT NOT NULL,  -- "ready" | "fetchable" | "private" | "unavailable"
    hf_uri           TEXT,
    source_url       TEXT,
    license          TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
```

---

## Environment Configuration

All configuration is read from `.env` at startup via `AppConfig::from_env()`.

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | _(required)_ | Authentication key for all endpoints |
| `PORT` | `9900` | HTTP server port |
| `DATABASE_URL` | `sqlite:guardrail.db` | SQLite path or Postgres URL |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API base (OpenAI-compatible) |
| `LLM_MODEL` | `gpt-4o-mini` | Model used for mirror augmentation + regex gen |
| `LLM_API_KEY` | _(required for LLM features)_ | LLM API key |
| `MAX_TEXT_CHARS` | `500000` | Max input text length in characters |
| `MIRROR_MAX_RECORDS` | `500` | Max LLM mirrors **per label class** per training call |
| `MODELS_DIR` | `./models` | Directory for base + custom weight files |
| `PARAPET_CONFIG` | `./parapet.yaml` | Parapet L3 pattern config path |
| `PYTHON_EXECUTABLE` | `python` | Python binary (set to conda path if needed) |
| `SCHEMA_EVAL_DIR` | `./schema/eval` | YAML dataset directory for catalog seeding |

> **LLM is opt-in per request.** Set `enable_mirror: true` in a training request to generate LLM mirrors. Set `use_llm: true` on pattern create/add to generate regex from plain text. Without these flags, no LLM calls are made.

---

## Training Pipeline

### Base Models (auto, startup)

`scripts/train_base_models.py` is called by `main.rs` if any base model `.weights.json` is missing. Trains 9 SVMs:

| Model | Training Data |
|---|---|
| `instruction_override` | HackAPrompt, ChatGPT Jailbreaks, curated mirror |
| `roleplay_jailbreak` | ChatGPT Jailbreaks, Jailbreak Cls, HackAPrompt |
| `meta_probe` | ChatGPT Jailbreaks, curated mirror |
| `exfiltration` | ChatGPT Jailbreaks, Jailbreak Cls |
| `adversarial_suffix` | Jailbreak Cls |
| `indirect_injection` | Jailbreak Cls |
| `obfuscation` | ChatGPT Jailbreaks, Jailbreak Cls, HackAPrompt |
| `constraint_bypass` | ChatGPT Jailbreaks, Jailbreak Cls, HackAPrompt |
| `allrounder` | All 8 category datasets combined |

### Custom Model Training (client-triggered)

`POST /v1/models/{id}/train` → spawns background `tokio::task`:

```
1. DELETE existing training_records for model_id  (clean retrain, prevents accumulation)
2. INSERT new client records into training_records
3. If enable_mirror=true:
     mirror_augment.py → LLM generates counterpart records
     Cap: MIRROR_MAX_RECORDS per label class (attack/benign independently)
     Summary JSON → logged by Rust
4. Export ALL training_records for model → temp JSONL
5. train_custom_model.py:
     a. Load JSONL (with full L0 pre-processing applied at load time)
     b. Load --blend-categories records (JSONL-cached per category)
     c. Load --blend-dataset-files (specific YAML files)
     d. SHA-256 global deduplication
     e. Stratified 85/15 train/holdout split
     f. Squash augmentation on TRAIN ONLY (prevents data leakage)
     g. Dynamic min_df: min_df=1 if train set < 50 records, else min_df=5
     h. LinearSVC(penalty="l1", dual=False, class_weight="balanced")
     i. Emit .weights.json + JSON metrics to stdout
6. Parse metrics → UPDATE custom_models:
     training_samples = total_samples (client + ALL blends, post-dedup)
     f1_score, status="ready"
```

### Key Training Fixes (v2 — 2026-07-27)

The following bugs were identified and fixed in `scripts/train_custom_model.py` by aligning with the legacy `train_l1_specialist.py` pipeline:

| Bug | Root Cause | Fix Applied |
|---|---|---|
| F1 = 1.0 (data leakage) | Squash augmentation happened before train/test split — identical clones in both sets | Squash augmentation now applied **after split, to train set only** |
| Inflated `training_samples` | Each `POST /train` appended records to DB; 4 runs × 6 records = 24 shown | Added `DELETE FROM training_records WHERE model_id = ?` before insert |
| Wrong count (blend not shown) | `training_samples` stored only client-submitted records, not blend datasets | Now stores `total_samples` (client + all blends, after dedup) |
| L2 penalty | Legacy uses L1 penalty for sparse features | Switched to `LinearSVC(penalty="l1", dual=False, class_weight="balanced")` |
| `min_df=1` always | Legacy uses `min_df=5` for general-purpose baseline | Dynamic: `min_df=1` if train set < 50 records, else `min_df=5` |
| No L0 at training time | Runtime L0 normalizes inputs but training did not | Full L0 preprocessing applied at JSONL load time in training |
| No deduplication | Duplicate records inflated apparent training size | SHA-256 content hash dedup before any split |

---

## Pattern Management (`scripts/generate_regex.py` + `src/api/patterns.rs`)

### LLM Opt-in (v2 — 2026-07-27)

LLM-assisted regex generation is **off by default**. It must be explicitly requested per API call:

- `POST /v1/patterns` — include `"use_llm": true` to enable LLM generation for plain-text inputs
- `POST /v1/patterns/{id}/entries` — same `"use_llm": true` flag

When `use_llm: false` (default):
- Valid regex → stored as-is (`source: "user_regex"`)
- Plain text that fails regex compilation → regex-escaped and stored as a literal match

When `use_llm: true`:
- Valid regex → stored as-is
- Plain text → sent to LLM → LLM returns patterns → stored as `source: "llm_generated"`
- If LLM fails → falls back to regex-escaped literal

Response includes `"llm_used": bool` so clients can confirm whether LLM was invoked.

---

## Key Design Decisions

### 1. Dynamic sqlx (no compile-time macros)
All DB queries use `sqlx::query()` / `sqlx::query_as()` (dynamic API). No `sqlx::query!()` macros. This eliminates the `DATABASE_URL`-at-compile-time requirement, enabling offline builds and runtime database configuration.

### 2. Independent Standalone Workspace
`parapet-guardrail/Cargo.toml` contains `[workspace]` and is unlinked from the parent workspace `members`. Builds create `parapet-guardrail/target/` — fully isolated.

### 3. Pre-compiled PHF Base Models
Base SVM models use compile-time PHF static maps (`phf::Map`) ported from legacy `l1_weights.rs`. Scoring is O(1) with zero startup I/O. Custom models still use dynamic `HashMap` loaded from `.weights.json` files.

### 4. Configurable Python Executable
`PYTHON_EXECUTABLE` env var controls which Python binary runs training/augmentation scripts. Supports Conda environments (e.g., `C:/Users/USER/.conda/envs/ml-guardrails/python.exe`). Defaults to `python`.

### 5. SQLite Auto-Create
`SqliteConnectOptions::create_if_missing(true)` so `guardrail.db` is created automatically on first run (no manual `sqlite3` command needed).

### 6. Axum 0.8 Path Syntax
All path parameters use `{id}` / `{entry_id}` syntax (not `:id`). Required by Axum 0.8 router.

### 7. Dataset Catalog with Startup Seeder
`src/datasets.rs` runs a non-blocking background task on startup that scans `SCHEMA_EVAL_DIR` YAML files, parses record counts, and upserts catalog entries into the DB. The catalog supports filtering by `category`, `status`, `label_type`, and `license`.

### 8. Mirror Cap per Label Class
`MIRROR_MAX_RECORDS` applies independently to attack records and benign records. With `MIRROR_MAX_RECORDS=500`, at most 500 attack mirrors AND 500 benign mirrors are generated per training call (max 1000 total LLM calls). Prevents token budget exhaustion on large training uploads.

---

## API Endpoints Summary

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/health` | None | Health check |
| `POST` | `/v1/detect` | ✅ | Score text with selected guardrails |
| `POST` | `/v1/patterns` | ✅ | Create pattern group (LLM opt-in via `use_llm`) |
| `GET` | `/v1/patterns` | ✅ | List pattern groups |
| `GET` | `/v1/patterns/{id}` | ✅ | Get group + entries |
| `PUT` | `/v1/patterns/{id}` | ✅ | Update name/description/category |
| `DELETE` | `/v1/patterns/{id}` | ✅ | Delete group + all entries |
| `POST` | `/v1/patterns/{id}/entries` | ✅ | Add patterns to existing group |
| `DELETE` | `/v1/patterns/{id}/entries/{entry_id}` | ✅ | Remove single entry |
| `POST` | `/v1/models` | ✅ | Register custom model |
| `GET` | `/v1/models` | ✅ | List custom models |
| `GET` | `/v1/models/{id}` | ✅ | Get model metadata |
| `DELETE` | `/v1/models/{id}` | ✅ | Delete model record + weight file |
| `POST` | `/v1/models/{id}/train` | ✅ | Trigger training (async) — LLM mirror opt-in |
| `GET` | `/v1/models/{id}/training-status` | ✅ | Poll training status |
| `GET` | `/v1/datasets` | ✅ | List dataset catalog with filters |
| `POST` | `/v1/datasets/{id}/fetch` | ✅ | Trigger on-demand dataset download |

---

## Quick Start

```powershell
# 1. Navigate to standalone directory
cd parapet-guardrail

# 2. Copy and configure environment
copy .env.example .env
# Edit .env: set API_KEY, LLM_API_KEY, PYTHON_EXECUTABLE (if using Conda)

# 3. Build and run
cargo run
# First run trains 9 base models automatically (takes several minutes)
# Subsequent runs skip training — model files are cached
```

See [api_developer_guide.md](api_developer_guide.md) for full API usage documentation.
