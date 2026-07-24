// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Database migrations — embedded SQL run on first connect.

use anyhow::Result;
use sqlx::{Pool, Postgres, Sqlite};

// ---------------------------------------------------------------------------
// Shared schema DDL (SQLite dialect — compatible subset)
// ---------------------------------------------------------------------------

const SCHEMA_SQLITE: &str = r#"
-- Pattern groups: one logical "pattern" entry with 1..N compiled regex strings.
CREATE TABLE IF NOT EXISTS pattern_groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    category    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Individual compiled regex patterns belonging to a pattern group.
CREATE TABLE IF NOT EXISTS pattern_entries (
    id          TEXT PRIMARY KEY,
    group_id    TEXT NOT NULL REFERENCES pattern_groups(id) ON DELETE CASCADE,
    raw_input   TEXT NOT NULL,
    pattern     TEXT NOT NULL,
    source      TEXT NOT NULL CHECK(source IN ('user_regex', 'llm_generated')),
    created_at  TEXT NOT NULL
);

-- Custom SVM models.
CREATE TABLE IF NOT EXISTS custom_models (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    description      TEXT,
    category         TEXT NOT NULL,
    status           TEXT NOT NULL CHECK(status IN ('pending', 'training', 'ready', 'error'))
                         DEFAULT 'pending',
    model_path       TEXT,
    training_samples INTEGER,
    f1_score         REAL,
    error_message    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- Training records stored per model.
CREATE TABLE IF NOT EXISTS training_records (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL REFERENCES custom_models(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    label         INTEGER NOT NULL CHECK(label IN (0, 1)),
    source        TEXT NOT NULL CHECK(source IN ('client', 'mirror_generated', 'base_blend')),
    mirror_of     TEXT,
    base_category TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pattern_entries_group_id ON pattern_entries(group_id);
CREATE INDEX IF NOT EXISTS idx_training_records_model_id ON training_records(model_id);
"#;

// ---------------------------------------------------------------------------
// Postgres dialect (slight differences in types)
// ---------------------------------------------------------------------------

const SCHEMA_POSTGRES: &str = r#"
CREATE TABLE IF NOT EXISTS pattern_groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    category    TEXT,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_entries (
    id          TEXT PRIMARY KEY,
    group_id    TEXT NOT NULL REFERENCES pattern_groups(id) ON DELETE CASCADE,
    raw_input   TEXT NOT NULL,
    pattern     TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_models (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    description      TEXT,
    category         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    model_path       TEXT,
    training_samples INTEGER,
    f1_score         DOUBLE PRECISION,
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS training_records (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL REFERENCES custom_models(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    label         SMALLINT NOT NULL,
    source        TEXT NOT NULL,
    mirror_of     TEXT,
    base_category TEXT,
    created_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pattern_entries_group_id ON pattern_entries(group_id);
CREATE INDEX IF NOT EXISTS idx_training_records_model_id ON training_records(model_id);
"#;

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

/// Run schema migrations on a SQLite pool.
pub async fn run_sqlite(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(SCHEMA_SQLITE).execute(pool).await?;
    tracing::info!("SQLite schema migrations applied");
    Ok(())
}

/// Run schema migrations on a Postgres pool.
pub async fn run_postgres(pool: &Pool<Postgres>) -> Result<()> {
    sqlx::query(SCHEMA_POSTGRES).execute(pool).await?;
    tracing::info!("Postgres schema migrations applied");
    Ok(())
}
