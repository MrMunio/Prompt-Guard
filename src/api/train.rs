// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Training API handlers.
//!
//! POST /v1/models/:id/train           — submit training records; starts async training
//! GET  /v1/models/:id/training-status — poll training status
//!
//! All DB queries use the dynamic sqlx API (no compile-time DATABASE_URL needed).

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use chrono::Utc;
use serde::Deserialize;
use sqlx::Row;
use uuid::Uuid;

use crate::api::models::load_model_response;
use crate::api::AppState;
use crate::db::DbPool;
use crate::engine::EngineState;
use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct TrainRequest {
    pub records: Vec<TrainingRecord>,
    #[serde(default)]
    pub blend_base_categories: BlendBaseInput,
}

#[derive(Deserialize)]
pub struct TrainingRecord {
    pub text: String,
    /// 0 = benign, 1 = attack
    pub label: u8,
}

#[derive(Deserialize, Default)]
#[serde(untagged)]
pub enum BlendBaseInput {
    All(crate::api::detect::AllTag),
    Categories(Vec<String>),
    #[default]
    None,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/models/:id/train
pub async fn train_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Json(req): Json<TrainRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let model = load_model_response(&model_id, &state.db).await?;

    if model.status == "training" {
        return Err(ApiError::Unprocessable(
            "Model is currently training. Wait for it to complete before submitting new records.".to_string()
        ));
    }

    if req.records.is_empty() {
        return Err(ApiError::bad("'records' must not be empty"));
    }
    if req.records.len() > 10_000 {
        return Err(ApiError::bad("'records' exceeds maximum of 10,000 per request"));
    }

    let mut field_errors = serde_json::Map::new();
    for (i, r) in req.records.iter().enumerate() {
        if r.text.trim().is_empty() {
            field_errors.insert(format!("records[{i}].text"), serde_json::json!("must not be empty"));
        }
        if r.label != 0 && r.label != 1 {
            field_errors.insert(format!("records[{i}].label"), serde_json::json!("must be 0 or 1"));
        }
    }
    if !field_errors.is_empty() {
        return Err(ApiError::bad_fields("Invalid training records", serde_json::Value::Object(field_errors)));
    }

    let now = Utc::now().to_rfc3339();
    for record in &req.records {
        insert_training_record(&model_id, &record.text, record.label as i64, &state.db, &now).await?;
    }

    // Mark model as 'training'.
    match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("UPDATE custom_models SET status='training', updated_at=? WHERE id=?")
                .bind(&now).bind(&model_id).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query("UPDATE custom_models SET status='training', updated_at=$1 WHERE id=$2")
                .bind(&now).bind(&model_id).execute(pool).await?;
        }
    }

    // Spawn background training task.
    let db_clone = state.db.clone();
    let engine_clone = state.engine.clone();
    let model_id_clone = model_id.clone();
    let models_dir = state.engine.models_dir.clone();
    let llm_base_url = state.engine.llm_base_url.clone();
    let llm_model = state.engine.llm_model.clone();
    let llm_api_key = state.engine.llm_api_key.clone();
    let blend_cats = match &req.blend_base_categories {
        BlendBaseInput::All(_) => vec!["all".to_string()],
        BlendBaseInput::Categories(cats) => cats.clone(),
        BlendBaseInput::None => vec![],
    };

    tokio::spawn(async move {
        run_training_pipeline(
            &model_id_clone, &models_dir, &llm_base_url, &llm_model,
            &llm_api_key, &blend_cats, &db_clone, &engine_clone,
        ).await;
    });

    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "model_id": model_id,
            "status": "training",
            "message": "Training started. Poll GET /v1/models/{id}/training-status for updates."
        }))
    ))
}

/// GET /v1/models/:id/training-status
pub async fn training_status_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let model = load_model_response(&id, &state.db).await?;
    Ok(Json(serde_json::json!({
        "model_id": model.id,
        "status": model.status,
        "training_samples": model.training_samples,
        "f1_score": model.f1_score,
        "error_message": model.error_message,
        "updated_at": model.updated_at,
    })))
}

// ---------------------------------------------------------------------------
// Background training pipeline
// ---------------------------------------------------------------------------

async fn run_training_pipeline(
    model_id: &str,
    models_dir: &str,
    llm_base_url: &str,
    llm_model: &str,
    llm_api_key: &str,
    blend_cats: &[String],
    db: &DbPool,
    engine: &EngineState,
) {
    if let Err(e) = do_training(model_id, models_dir, llm_base_url, llm_model,
                                llm_api_key, blend_cats, db, engine).await {
        tracing::error!(model_id, error = %e, "Training failed");
        let now = Utc::now().to_rfc3339();
        let msg = e.to_string();
        let _ = set_model_error(model_id, &msg, &now, db).await;
    }
}

async fn set_model_error(model_id: &str, msg: &str, now: &str, db: &DbPool) -> anyhow::Result<()> {
    match db {
        DbPool::Sqlite(pool) => {
            sqlx::query("UPDATE custom_models SET status='error', error_message=?, updated_at=? WHERE id=?")
                .bind(msg).bind(now).bind(model_id).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query("UPDATE custom_models SET status='error', error_message=$1, updated_at=$2 WHERE id=$3")
                .bind(msg).bind(now).bind(model_id).execute(pool).await?;
        }
    }
    Ok(())
}

async fn do_training(
    model_id: &str,
    models_dir: &str,
    llm_base_url: &str,
    llm_model: &str,
    llm_api_key: &str,
    blend_cats: &[String],
    db: &DbPool,
    engine: &EngineState,
) -> anyhow::Result<()> {
    // 1. Mirror augmentation.
    tracing::info!(model_id, "Starting mirror augmentation");
    let db_url = match db { DbPool::Sqlite(_) => "sqlite:guardrail.db", DbPool::Postgres(_) => "postgres" };
    let aug_output = tokio::process::Command::new(&engine.python_executable)
        .arg("scripts/mirror_augment.py")
        .arg("--model-id").arg(model_id)
        .arg("--db-url").arg(db_url)
        .arg("--base-url").arg(llm_base_url)
        .arg("--model").arg(llm_model)
        .arg("--api-key").arg(llm_api_key)
        .output().await?;

    if !aug_output.status.success() {
        let err = String::from_utf8_lossy(&aug_output.stderr);
        anyhow::bail!("mirror_augment.py failed: {err}");
    }

    // 2. Export training records to JSONL.
    let data_path = format!("{models_dir}/custom/{model_id}_train.jsonl");
    export_training_records(model_id, &data_path, db).await?;

    // 3. Run training script.
    let weights_path = format!("{models_dir}/custom/{model_id}.weights.json");
    let schema_dir = "./schema/eval";
    let cache_dir = format!("{models_dir}/base_cache");
    tracing::info!(model_id, blend = ?blend_cats, "Starting SVM training");
    let mut cmd = tokio::process::Command::new(&engine.python_executable);
    cmd.arg("scripts/train_custom_model.py")
        .arg("--data-file").arg(&data_path)
        .arg("--out-weights").arg(&weights_path)
        .arg("--schema-dir").arg(schema_dir)
        .arg("--cache-dir").arg(&cache_dir);
    // Append blend categories if any were requested.
    if !blend_cats.is_empty() {
        cmd.arg("--blend-categories").args(blend_cats);
    }
    let train_output = cmd.output().await?;

    if !train_output.status.success() {
        let err = String::from_utf8_lossy(&train_output.stderr);
        anyhow::bail!("train_custom_model.py failed: {err}");
    }

    // 4. Parse metrics.
    let metrics_str = String::from_utf8_lossy(&train_output.stdout);
    let metrics: serde_json::Value = serde_json::from_str(&metrics_str).unwrap_or(serde_json::json!({}));
    let f1 = metrics["f1"].as_f64();
    let samples = metrics["samples"].as_i64();
    let blend_samples = metrics["blend_samples"].as_i64().unwrap_or(0);
    tracing::info!(model_id, f1 = ?f1, client_samples = ?samples, blend_samples, "Training metrics");

    // 5. Update model record.
    let now = Utc::now().to_rfc3339();
    match db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "UPDATE custom_models SET status='ready', model_path=?, f1_score=?, training_samples=?, updated_at=? WHERE id=?"
            ).bind(&weights_path).bind(f1).bind(samples).bind(&now).bind(model_id)
             .execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "UPDATE custom_models SET status='ready', model_path=$1, f1_score=$2, training_samples=$3, updated_at=$4 WHERE id=$5"
            ).bind(&weights_path).bind(f1).bind(samples).bind(&now).bind(model_id)
             .execute(pool).await?;
        }
    }

    // 6. Evict from cache.
    engine.custom_models.evict(model_id).await;

    // 7. Clean up temp data file.
    let _ = std::fs::remove_file(&data_path);

    tracing::info!(model_id, f1 = ?f1, samples = ?samples, "Training complete");
    Ok(())
}

async fn insert_training_record(
    model_id: &str,
    text: &str,
    label: i64,
    db: &DbPool,
    now: &str,
) -> Result<(), ApiError> {
    let record_id = Uuid::new_v4().to_string();
    match db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "INSERT INTO training_records (id, model_id, text, label, source, created_at) VALUES (?,?,?,?,'client',?)"
            ).bind(&record_id).bind(model_id).bind(text).bind(label).bind(now)
             .execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO training_records (id, model_id, text, label, source, created_at) VALUES ($1,$2,$3,$4,'client',$5)"
            ).bind(&record_id).bind(model_id).bind(text).bind(label).bind(now)
             .execute(pool).await?;
        }
    }
    Ok(())
}

async fn export_training_records(
    model_id: &str,
    output_path: &str,
    db: &DbPool,
) -> anyhow::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(
        std::path::Path::new(output_path).parent().unwrap_or(std::path::Path::new("."))
    )?;
    let mut file = std::fs::File::create(output_path)?;

    // Export all client + mirror-generated records for this model to JSONL.
    // Base corpus blending is handled by train_custom_model.py via --blend-categories.
    let records: Vec<(String, i64)> = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query("SELECT text, label FROM training_records WHERE model_id = ?")
                .bind(model_id).fetch_all(pool).await?
                .iter().map(|r| (r.get::<String, _>("text"), r.get::<i64, _>("label"))).collect()
        }
        DbPool::Postgres(pool) => {
            sqlx::query("SELECT text, label FROM training_records WHERE model_id = $1")
                .bind(model_id).fetch_all(pool).await?
                .iter().map(|r| (r.get::<String, _>("text"), r.get::<i64, _>("label"))).collect()
        }
    };

    for (text, label) in records {
        let line = serde_json::json!({ "text": text, "label": label });
        writeln!(file, "{line}")?;
    }

    Ok(())
}
