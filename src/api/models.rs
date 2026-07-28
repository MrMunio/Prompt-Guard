// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Custom model CRUD API handlers.
//!
//! POST   /v1/models       — register a new custom model
//! GET    /v1/models       — list all custom models
//! GET    /v1/models/:id   — get model metadata
//! DELETE /v1/models/:id   — delete model record + weights file
//!
//! All DB queries use the dynamic sqlx API (no compile-time DATABASE_URL needed).

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::api::AppState;
use crate::db::DbPool;
use crate::engine::svm_base::is_valid_base_category;
use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateModelRequest {
    pub id: Option<String>,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub status: String,
    pub model_path: Option<String>,
    pub training_samples: Option<i64>,
    pub f1_score: Option<f64>,
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Row → ModelResponse helper
// ---------------------------------------------------------------------------

fn row_to_model(row: &sqlx::sqlite::SqliteRow) -> ModelResponse {
    use sqlx::Row as _;
    ModelResponse {
        id: row.get("id"),
        name: row.get("name"),
        description: row.get("description"),
        category: row.get("category"),
        status: row.get("status"),
        model_path: row.get("model_path"),
        training_samples: row.get("training_samples"),
        f1_score: row.get("f1_score"),
        error_message: row.get("error_message"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

fn pg_row_to_model(row: &sqlx::postgres::PgRow) -> ModelResponse {
    use sqlx::Row as _;
    ModelResponse {
        id: row.get("id"),
        name: row.get("name"),
        description: row.get("description"),
        category: row.get("category"),
        status: row.get("status"),
        model_path: row.get("model_path"),
        training_samples: row.get("training_samples"),
        f1_score: row.get("f1_score"),
        error_message: row.get("error_message"),
        created_at: row.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
        updated_at: row.get::<chrono::DateTime<chrono::Utc>, _>("updated_at").to_rfc3339(),
    }
}

const SELECT_MODEL: &str =
    "SELECT id, name, description, category, status, model_path, \
     training_samples, f1_score, error_message, created_at, updated_at \
     FROM custom_models";

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/models
pub async fn create_model(
    State(state): State<AppState>,
    Json(req): Json<CreateModelRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::bad("'name' must not be empty"));
    }
    if !is_valid_base_category(&req.category) && req.category != "custom" {
        return Err(ApiError::bad_fields(
            "Invalid 'category' — must be one of the 8 canonical attack categories or 'custom'",
            serde_json::json!({ "category": req.category }),
        ));
    }

    let model_id = req.id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let now = Utc::now().to_rfc3339();

    check_model_id_available(&model_id, &state.db).await?;
    check_model_name_available(&req.name, &state.db).await?;

    match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "INSERT INTO custom_models (id, name, description, category, status, created_at, updated_at)
                 VALUES (?, ?, ?, ?, 'pending', ?, ?)"
            ).bind(&model_id).bind(&req.name).bind(&req.description).bind(&req.category)
             .bind(&now).bind(&now).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO custom_models (id, name, description, category, status, created_at, updated_at)
                 VALUES ($1, $2, $3, $4, 'pending', $5, $6)"
            ).bind(&model_id).bind(&req.name).bind(&req.description).bind(&req.category)
             .bind(&now).bind(&now).execute(pool).await?;
        }
    }

    let resp = load_model_response(&model_id, &state.db).await?;
    Ok((axum::http::StatusCode::CREATED, Json(resp)))
}

#[derive(Debug, Serialize, Clone)]
pub struct BaseSvmModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
}

pub fn get_base_models_info() -> Vec<BaseSvmModelInfo> {
    vec![
        BaseSvmModelInfo {
            id: "allrounder".to_string(),
            name: "Allrounder Composite SVM".to_string(),
            description: "Combined multi-category classifier trained on all attack categories".to_string(),
            category: "allrounder".to_string(),
        },
        BaseSvmModelInfo {
            id: "instruction_override".to_string(),
            name: "Instruction Override SVM".to_string(),
            description: "Detects commands attempting to ignore, override, or replace system instructions".to_string(),
            category: "instruction_override".to_string(),
        },
        BaseSvmModelInfo {
            id: "roleplay_jailbreak".to_string(),
            name: "Roleplay Jailbreak SVM".to_string(),
            description: "Detects role-play and persona switching techniques used to bypass safety restrictions".to_string(),
            category: "roleplay_jailbreak".to_string(),
        },
        BaseSvmModelInfo {
            id: "meta_probe".to_string(),
            name: "Meta Probe SVM".to_string(),
            description: "Detects questions probing system prompt details, instructions, or internal identity".to_string(),
            category: "meta_probe".to_string(),
        },
        BaseSvmModelInfo {
            id: "exfiltration".to_string(),
            name: "Exfiltration SVM".to_string(),
            description: "Detects attempts to extract confidential system instructions, files, or sensitive data".to_string(),
            category: "exfiltration".to_string(),
        },
        BaseSvmModelInfo {
            id: "adversarial_suffix".to_string(),
            name: "Adversarial Suffix SVM".to_string(),
            description: "Detects adversarial noise or token suffixes appended to trick safety classifiers".to_string(),
            category: "adversarial_suffix".to_string(),
        },
        BaseSvmModelInfo {
            id: "indirect_injection".to_string(),
            name: "Indirect Injection SVM".to_string(),
            description: "Detects prompt injection embedded within external documents, web pages, or tool outputs".to_string(),
            category: "indirect_injection".to_string(),
        },
        BaseSvmModelInfo {
            id: "obfuscation".to_string(),
            name: "Obfuscation SVM".to_string(),
            description: "Detects encoding tricks, leetspeak, or invisible characters used to obscure malicious intent".to_string(),
            category: "obfuscation".to_string(),
        },
        BaseSvmModelInfo {
            id: "constraint_bypass".to_string(),
            name: "Constraint Bypass SVM".to_string(),
            description: "Detects requests urging the model to relax, re-interpret, or ignore safety boundaries".to_string(),
            category: "constraint_bypass".to_string(),
        },
    ]
}

/// GET /v1/models
pub async fn list_models(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let sql = format!("{SELECT_MODEL} ORDER BY created_at DESC");
    let rows = match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query(&sql).fetch_all(pool).await?
                .iter().map(row_to_model).collect::<Vec<_>>()
        }
        DbPool::Postgres(pool) => {
            sqlx::query(&sql).fetch_all(pool).await?
                .iter().map(pg_row_to_model).collect::<Vec<_>>()
        }
    };
    let base_models = get_base_models_info();
    Ok(Json(serde_json::json!({
        "base": base_models,
        "custom": rows,
    })))
}

/// GET /v1/models/:id
pub async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let resp = load_model_response(&id, &state.db).await?;
    Ok(Json(resp))
}

/// DELETE /v1/models/:id
pub async fn delete_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    // Get model_path before deleting.
    let model_path: Option<String> = match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("SELECT model_path FROM custom_models WHERE id = ?")
                .bind(&id).fetch_optional(pool).await?
                .map(|r: sqlx::sqlite::SqliteRow| r.get("model_path"))
        }
        DbPool::Postgres(pool) => {
            sqlx::query("SELECT model_path FROM custom_models WHERE id = $1")
                .bind(&id).fetch_optional(pool).await?
                .map(|r: sqlx::postgres::PgRow| r.get("model_path"))
        }
    };

    // Check it exists.
    if model_path.is_none() {
        // still try delete — if row missing, 404
        let exists = check_model_exists(&id, &state.db).await?;
        if !exists {
            return Err(ApiError::NotFound(format!("Model '{id}' not found")));
        }
    }

    match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("DELETE FROM custom_models WHERE id = ?")
                .bind(&id).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query("DELETE FROM custom_models WHERE id = $1")
                .bind(&id).execute(pool).await?;
        }
    }

    // Remove weights file.
    if let Some(Some(path)) = Some(model_path).filter(|p| p.is_some()) {
        if let Err(e) = std::fs::remove_file(&path) {
            tracing::warn!(path, error = %e, "Could not remove model weights file");
        }
    }

    // Evict from cache.
    state.engine.custom_models.evict(&id).await;

    Ok(axum::http::StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn check_model_exists(id: &str, db: &DbPool) -> Result<bool, ApiError> {
    let count: i64 = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query("SELECT COUNT(*) as count FROM custom_models WHERE id = ?")
                .bind(id).fetch_one(pool).await?
                .get::<i64, _>("count")
        }
        DbPool::Postgres(pool) => {
            sqlx::query("SELECT COUNT(*) as count FROM custom_models WHERE id = $1")
                .bind(id).fetch_one(pool).await?
                .get::<i64, _>("count")
        }
    };
    Ok(count > 0)
}

async fn check_model_id_available(id: &str, db: &DbPool) -> Result<(), ApiError> {
    if check_model_exists(id, db).await? {
        Err(ApiError::Conflict(format!("Model id '{id}' is already taken")))
    } else {
        Ok(())
    }
}

async fn check_model_name_available(name: &str, db: &DbPool) -> Result<(), ApiError> {
    let count: i64 = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query("SELECT COUNT(*) as count FROM custom_models WHERE name = ?")
                .bind(name).fetch_one(pool).await?
                .get::<i64, _>("count")
        }
        DbPool::Postgres(pool) => {
            sqlx::query("SELECT COUNT(*) as count FROM custom_models WHERE name = $1")
                .bind(name).fetch_one(pool).await?
                .get::<i64, _>("count")
        }
    };
    if count > 0 {
        Err(ApiError::Conflict(format!("Model name '{name}' is already taken")))
    } else {
        Ok(())
    }
}

pub async fn load_model_response(id: &str, db: &DbPool) -> Result<ModelResponse, ApiError> {
    match db {
        DbPool::Sqlite(pool) => {
            let sql = format!("{SELECT_MODEL} WHERE id = ?");
            sqlx::query(&sql).bind(id).fetch_optional(pool).await?
                .map(|r| row_to_model(&r))
                .ok_or_else(|| ApiError::NotFound(format!("Model '{id}' not found")))
        }
        DbPool::Postgres(pool) => {
            let sql = format!("{SELECT_MODEL} WHERE id = $1");
            sqlx::query(&sql).bind(id).fetch_optional(pool).await?
                .map(|r| pg_row_to_model(&r))
                .ok_or_else(|| ApiError::NotFound(format!("Model '{id}' not found")))
        }
    }
}
