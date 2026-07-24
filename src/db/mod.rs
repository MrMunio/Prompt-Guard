// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Database abstraction layer.
//!
//! Uses dynamic sqlx queries (no compile-time DATABASE_URL required).
//! Pool is detected at runtime from `DATABASE_URL` env var.

pub mod migrations;

use anyhow::Result;
use sqlx::{Pool, Postgres, Sqlite};

// ---------------------------------------------------------------------------
// Pool union type
// ---------------------------------------------------------------------------

/// Runtime-detected database pool. Wraps either SQLite or Postgres.
#[derive(Clone, Debug)]
pub enum DbPool {
    Sqlite(Pool<Sqlite>),
    Postgres(Pool<Postgres>),
}

impl DbPool {
    /// Connect to the database specified by `database_url` and run migrations.
    pub async fn connect(database_url: &str) -> Result<Self> {
        if database_url.starts_with("sqlite") {
            tracing::info!(url = database_url, "connecting to SQLite");
            use std::str::FromStr;
            let opts = sqlx::sqlite::SqliteConnectOptions::from_str(database_url)?
                .create_if_missing(true);
            let pool = sqlx::SqlitePool::connect_with(opts).await?;
            migrations::run_sqlite(&pool).await?;
            Ok(DbPool::Sqlite(pool))
        } else if database_url.starts_with("postgres") {
            tracing::info!("connecting to Postgres");
            let pool = sqlx::PgPool::connect(database_url).await?;
            migrations::run_postgres(&pool).await?;
            Ok(DbPool::Postgres(pool))
        } else {
            anyhow::bail!(
                "Unsupported DATABASE_URL scheme. Use 'sqlite:...' or 'postgres://...'."
            )
        }
    }

    #[allow(dead_code)]
    pub fn is_sqlite(&self) -> bool {
        matches!(self, DbPool::Sqlite(_))
    }
}
