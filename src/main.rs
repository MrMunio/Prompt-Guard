// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! parapet-guardrail — flexible prompt injection guardrail API.
//!
//! Startup sequence:
//!   1. Load configuration from environment
//!   2. Connect to database + run migrations
//!   3. Check base model cache; train missing models if needed
//!   4. Load base models into memory
//!   5. Build EngineState (GuardrailEngine + config values)
//!   6. Start axum HTTP server

mod api;
mod auth;
mod config;
mod db;
mod engine;
mod error;

use std::sync::Arc;

use tracing_subscriber::EnvFilter;

use crate::api::{build_router, AppState};
use crate::config::AppConfig;
use crate::db::DbPool;
use crate::engine::{EngineState, GuardrailEngine};
use crate::engine::svm_base::{BaseModelRegistry, BASE_MODEL_NAMES};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    tracing::info!("parapet-guardrail starting");

    // 1. Config.
    let config = AppConfig::from_env();
    tracing::info!(port = config.port, "configuration loaded");

    // 2. Database.
    let db = DbPool::connect(&config.database_url).await?;
    let db = Arc::new(db);

    // 3. Base model cache check and auto-training.
    ensure_base_models(&config.models_dir, &config.python_executable).await?;

    // 4. Load base models into memory.
    let base_models = BaseModelRegistry::load(&config.models_dir)?;

    // 5. Build engine + state.
    let guardrail_engine = GuardrailEngine::new(
        base_models,
        &config.parapet_config,
        &config.models_dir,
    )?;
    let engine_state = Arc::new(EngineState {
        engine: guardrail_engine,
        max_text_chars: config.max_text_chars,
        llm_base_url: config.llm_base_url.clone(),
        llm_model: config.llm_model.clone(),
        llm_api_key: config.llm_api_key.clone(),
        python_executable: config.python_executable.clone(),
    });

    let state = AppState {
        db,
        engine: engine_state,
    };

    // 6. Build router + start server.
    let router = build_router(state, config.api_key.clone());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", config.port)).await?;
    tracing::info!(port = config.port, "listening");
    axum::serve(listener, router).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Base model auto-training
// ---------------------------------------------------------------------------

/// Checks if all 9 base model weight files are present. If any are missing,
/// runs `scripts/train_base_models.py` to generate them (blocking startup).
async fn ensure_base_models(models_dir: &str, python_bin: &str) -> anyhow::Result<()> {
    let base_dir = std::path::Path::new(models_dir).join("base");
    std::fs::create_dir_all(&base_dir)?;

    let missing: Vec<&str> = BASE_MODEL_NAMES
        .iter()
        .filter(|&&name| !base_dir.join(format!("{name}.weights.json")).exists())
        .copied()
        .collect();

    if missing.is_empty() {
        tracing::info!("All 9 base models found in cache — skipping training");
        return Ok(());
    }

    tracing::warn!(
        missing = ?missing,
        python_bin = %python_bin,
        "Missing base models — running train_base_models.py (this may take several minutes)"
    );

    let status = tokio::process::Command::new(python_bin)
        .arg("scripts/train_base_models.py")
        .arg("--models-dir")
        .arg(models_dir)
        .status()
        .await?;

    if !status.success() {
        anyhow::bail!(
            "train_base_models.py failed (exit code {:?}). \
             Check Python dependencies: pip install scikit-learn numpy pyyaml",
            status.code()
        );
    }

    tracing::info!("Base model training complete");
    Ok(())
}
