# Parapet Guardrail Engine (`parapet-guardrail`)

> **Flexible, operator-deployable prompt-injection guardrail API.**  
> Self-contained HTTP service built in Rust (Axum) that wraps LLM calls with SVM classification, regex scanning, and L0 normalization — without touching your existing LLM client code.

---

## Overview

`parapet-guardrail` is a standalone prompt-injection guardrail API providing:

- **Per-request text scoring** via configurable SVM classifiers and regex scanners
- **9 pre-compiled base SVM models** (auto-loaded/trained on startup)
- **Client-registered custom SVM models** trained via REST API (async, background)
- **Client-registered custom regex pattern groups** (LLM-assisted, opt-in)
- **L0 normalization** on every request (NFKC, HTML strip, zero-width removal)
- **Single-key API authentication** (constant-time comparison)
- **SQLite (dev) and Postgres (prod)** with embedded auto-migrations
- **Dataset catalog** for blending open-source training data into custom models

---

## Architecture & Directory Structure

```
parapet-guardrail/
├── README.md                            ← Project overview & documentation
├── .env / .env.example                  ← Environment configuration
├── .gitignore                           ← Standard ignores (env, db, target, schema/eval)
├── Cargo.toml                           ← Independent workspace (standalone)
├── Dockerfile                           ← Multi-stage: Rust + Python runtime
├── docker-compose.yml                   ← Dev/prod compose with volumes
├── parapet.yaml                         ← Local parapet L3 pattern config
├── requirements.txt                     ← Python dependencies for ML scripts
│
├── schema/                              ← Dataset YAML files for base models
├── models/
│   ├── base/                            ← 9 base SVM weight files
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
│   └── base_cache/                      ← Per-category JSONL blend caches
│       ├── _benign.jsonl
│       └── {category}.jsonl
│
├── src/
│   ├── main.rs                          ← Startup orchestration + axum server
│   ├── config.rs                        ← AppConfig from env vars
│   ├── auth.rs                          ← X-API-Key middleware (constant-time)
│   ├── error.rs                         ← ApiError → HTTP responses
│   ├── datasets.rs                      ← Startup dataset catalog seeder
│   ├── db/
│   │   ├── mod.rs                       ← DbPool enum: Sqlite | Postgres
│   │   └── migrations.rs                ← Embedded SQL schema (auto-runs on connect)
│   ├── engine/
│   │   ├── mod.rs                       ← GuardrailEngine + EngineState (shared state)
│   │   ├── l0.rs                        ← L0 normalization wrapper
│   │   ├── scoring.rs                   ← N-gram tokenization + SVM dot-product
│   │   ├── svm_base.rs                  ← BaseModelRegistry (9 base weights)
│   │   ├── svm_custom.rs                ← CustomModelCache (lazy-load + LRU eviction)
│   │   ├── regex_base.rs                ← Built-in L3 scanner (DefaultInboundScanner)
│   │   ├── regex_custom.rs              ← Custom pattern group cache + regex scan
│   │   └── verdict.rs                   ← DetectResponse + GuardrailResult types
│   └── api/
│       ├── mod.rs                       ← Router: Axum 0.8 {id} path syntax + auth
│       ├── health.rs                    ← GET /v1/health
│       ├── detect.rs                    ← POST /v1/detect
│       ├── patterns.rs                  ← CRUD /v1/patterns
│       ├── models.rs                    ← CRUD /v1/models
│       ├── datasets.rs                  ← GET /v1/datasets + POST /v1/datasets/{id}/fetch
│       └── train.rs                     ← POST /v1/models/{id}/train + status
│
└── scripts/
    ├── train_base_models.py             ← Train all 9 base SVMs from schema/eval/ YAML files
    ├── train_custom_model.py            ← Train custom SVM (L0 preprocessing + deduplication)
    ├── mirror_augment.py                ← LLM mirror augmentation (Mirror Pattern)
    ├── generate_regex.py                ← LLM regex generation from descriptions
    └── sources/                         ← Dataset fetch scripts
```

---

## Request Pipeline

Every detection request flows through a 4-stage pipeline:

```
Client HTTP Request
       │
       ▼
┌─────────────────┐
│   auth.rs       │  X-API-Key header check (constant-time)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   engine/l0.rs  │  L0 Normalization (always):
│                 │  HTML strip → zero-width removal → NFKC → control char cleanup
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Scoring                             │
│                                                                 │
│  svm_base.rs       svm_custom.rs       regex_base.rs           │
│  Precompiled map   LRU cached          parapet L3 scanner       │
│                                                                 │
│                    regex_custom.rs                              │
│                    DB-stored regex     compiled on demand        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   verdict.rs    │  Aggregate scores & verdicts
                    └────────┬────────┘
                             │
                             ▼
                    JSON Verdict Response
```

---

## Base SVM Models

| Model ID | Description |
|---|---|
| `allrounder` | Trained on all 8 attack categories combined (default for `"svm_base": "all"`) |
| `instruction_override` | Instruction hijacking / override attempts |
| `roleplay_jailbreak` | Roleplay and persona-based jailbreaks |
| `meta_probe` | System prompt probing / meta-questioning |
| `exfiltration` | Data exfiltration via prompt manipulation |
| `adversarial_suffix` | Optimised adversarial token suffixes |
| `indirect_injection` | Injection via documents / tool results |
| `obfuscation` | Encoding and obfuscation-based evasion |
| `constraint_bypass` | Policy and safety-filter bypass |

---

## Quick Start

### 1. Configure Environment

```bash
cd parapet-guardrail
cp .env.example .env
# Edit .env — set API_KEY, LLM_API_KEY, PYTHON_EXECUTABLE, etc.
```

### 2. Build and Run

```bash
# Standalone Cargo run
cargo run

# Or with Docker
docker compose up --build
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | *(required)* | Authentication key for all endpoints |
| `PORT` | `9900` | HTTP server port |
| `DATABASE_URL` | `sqlite:guardrail.db` | SQLite path or Postgres URL |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Model for mirror augmentation & regex gen |
| `LLM_API_KEY` | *(required for LLM)* | LLM API key |
| `MAX_TEXT_CHARS` | `500000` | Max input text length in characters |
| `MIRROR_MAX_RECORDS` | `500` | Max LLM mirrors per label class per training call |
| `MODELS_DIR` | `./models` | Directory for base & custom weight files |
| `PARAPET_CONFIG` | `./parapet.yaml` | Parapet L3 pattern config path |
| `PYTHON_EXECUTABLE` | `python` | Python binary path |
| `SCHEMA_EVAL_DIR` | `./schema/eval` | Dataset directory for catalog seeding |

---

## API Endpoints Summary

All endpoints except `GET /v1/health` require the `X-API-Key` header.

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
n every input unconditionally — confusable characters cannot bypass SVM scoring
- LLM API key is never logged
- Custom model weight files are stored server-side only; clients receive metrics (F1, sample count) but never raw weights
- SQLite file and model weights directory should be excluded from version control (see `.gitignore`)
