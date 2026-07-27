// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Dataset catalog API handlers.
//!
//! GET  /v1/datasets              — list catalog with optional ?category= and ?status= filters
//! POST /v1/datasets/:id/fetch    — trigger on-demand download of a fetchable dataset

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use sqlx::Row;

use crate::api::AppState;
use crate::db::DbPool;
use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct DatasetEntry {
    pub id: String,
    pub file_name: String,
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub category: String,
    pub label_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attack_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benign_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    pub fetch_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

// ---------------------------------------------------------------------------
// Query params
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct DatasetQuery {
    pub category: Option<String>,
    /// Filter by fetch_status: "ready" | "fetchable" | "private" | "unavailable"
    pub status: Option<String>,
    /// Filter by label_type: "attack_only" | "benign_only" | "mixed"
    pub label_type: Option<String>,
    /// Filter by license string (e.g., "apache-2.0", "cc-by-4.0", "mit")
    pub license: Option<String>,
}

// ---------------------------------------------------------------------------
// GET /v1/datasets
// ---------------------------------------------------------------------------

pub async fn list_datasets_handler(
    State(state): State<AppState>,
    Query(query): Query<DatasetQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let rows = fetch_datasets(
        &state.db,
        query.category.as_deref(),
        query.status.as_deref(),
        query.label_type.as_deref(),
        query.license.as_deref(),
    ).await?;

    // Build list of supported/unavailable categories for summary.
    let all_categories = [
        "instruction_override", "roleplay_jailbreak", "meta_probe", "exfiltration",
        "adversarial_suffix", "indirect_injection", "obfuscation", "constraint_bypass",
    ];
    let ready_categories: std::collections::HashSet<&str> = rows.iter()
        .filter(|r| r.fetch_status == "ready")
        .map(|r| r.category.as_str())
        .collect();

    let categories_supported: Vec<&str> = all_categories.iter()
        .filter(|&&c| ready_categories.contains(c))
        .copied()
        .collect();
    let categories_unavailable: Vec<&str> = all_categories.iter()
        .filter(|&&c| !ready_categories.contains(c))
        .copied()
        .collect();

    let total = rows.len();
    Ok(Json(serde_json::json!({
        "datasets": rows,
        "total": total,
        "categories_supported": categories_supported,
        "categories_unavailable": categories_unavailable,
    })))
}

// ---------------------------------------------------------------------------
// POST /v1/datasets/:id/fetch
// ---------------------------------------------------------------------------

pub async fn fetch_dataset_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    // Look up the dataset.
    let row: Option<(String, String)> = match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("SELECT id, fetch_status FROM training_datasets WHERE id = ?")
                .bind(&id)
                .fetch_optional(pool)
                .await?
                .map(|r| (r.get::<String, _>("id"), r.get::<String, _>("fetch_status")))
        }
        DbPool::Postgres(pool) => {
            sqlx::query("SELECT id, fetch_status FROM training_datasets WHERE id = $1")
                .bind(&id)
                .fetch_optional(pool)
                .await?
                .map(|r| (r.get::<String, _>("id"), r.get::<String, _>("fetch_status")))
        }
    };

    let (_ds_id, fetch_status) = row.ok_or_else(|| {
        ApiError::NotFound(format!("Dataset '{id}' not found. Use GET /v1/datasets to list available datasets."))
    })?;
    match fetch_status.as_str() {
        "ready" => {
            return Ok(Json(serde_json::json!({
                "id": id,
                "status": "already_ready",
                "message": "Dataset is already available for blending."
            })));
        }
        "private" => {
            return Err(ApiError::Unprocessable(
                format!("Dataset '{id}' is private and cannot be fetched automatically.")
            ));
        }
        "unavailable" => {
            return Err(ApiError::Unprocessable(
                format!("Dataset '{id}' has no fetch script and is marked unavailable.")
            ));
        }
        _ => {} // fetchable — proceed
    }

    // For now, return a 202 with instructions since the actual fetch scripts
    // run as Python processes. A future iteration will spawn the fetch script.
    Ok(Json(serde_json::json!({
        "id": id,
        "status": "fetch_queued",
        "message": format!(
            "Dataset '{}' fetch acknowledged. \
             Run 'python scripts/sources/fetch_{}.py' from the project root, \
             then restart the server to re-index.",
            id, id.replace("opensource_", "").replace("_attacks", "").replace("_benign", "")
        )
    })))
}

// ---------------------------------------------------------------------------
// Internal: query helper
// ---------------------------------------------------------------------------

async fn fetch_datasets(
    db: &DbPool,
    category: Option<&str>,
    status: Option<&str>,
    label_type: Option<&str>,
    license: Option<&str>,
) -> Result<Vec<DatasetEntry>, ApiError> {
    macro_rules! map_row {
        ($row:expr) => {
            DatasetEntry {
                id:           $row.try_get("id").unwrap_or_default(),
                file_name:    $row.try_get("file_name").unwrap_or_default(),
                display_name: $row.try_get("display_name").unwrap_or_default(),
                description:  $row.try_get("description").ok(),
                category:     $row.try_get("category").unwrap_or_default(),
                label_type:   $row.try_get("label_type").unwrap_or_default(),
                record_count: $row.try_get("record_count").ok(),
                attack_count: $row.try_get("attack_count").ok(),
                benign_count: $row.try_get("benign_count").ok(),
                file_path:    $row.try_get("file_path").ok(),
                fetch_status: $row.try_get("fetch_status").unwrap_or_default(),
                hf_uri:       $row.try_get("hf_uri").ok(),
                source_url:   $row.try_get("source_url").ok(),
                license:      $row.try_get("license").ok(),
            }
        };
    }

    match db {
        DbPool::Sqlite(pool) => {
            // Build dynamic WHERE clause for SQLite.
            let mut conditions = Vec::new();
            if category.is_some()   { conditions.push("category = ?"); }
            if status.is_some()     { conditions.push("fetch_status = ?"); }
            if label_type.is_some() { conditions.push("label_type = ?"); }
            if license.is_some()    { conditions.push("license = ?"); }
            let where_clause = if conditions.is_empty() {
                String::new()
            } else {
                format!("WHERE {}", conditions.join(" AND "))
            };
            let sql = format!(
                "SELECT id, file_name, display_name, description, category, label_type, \
                 record_count, attack_count, benign_count, file_path, fetch_status, \
                 hf_uri, source_url, license \
                 FROM training_datasets {} ORDER BY category, id",
                where_clause
            );
            let mut q = sqlx::query(&sql);
            if let Some(c) = category   { q = q.bind(c); }
            if let Some(s) = status     { q = q.bind(s); }
            if let Some(l) = label_type { q = q.bind(l); }
            if let Some(li) = license   { q = q.bind(li); }
            let rows = q.fetch_all(pool).await?;
            Ok(rows.iter().map(|r| map_row!(r)).collect())
        }
        DbPool::Postgres(pool) => {
            let mut conditions = Vec::new();
            let mut idx = 1usize;
            if category.is_some()   { conditions.push(format!("category = ${idx}")); idx += 1; }
            if status.is_some()     { conditions.push(format!("fetch_status = ${idx}")); idx += 1; }
            if label_type.is_some() { conditions.push(format!("label_type = ${idx}")); idx += 1; }
            if license.is_some()    { conditions.push(format!("license = ${idx}")); idx += 1; }
            let _ = idx; // suppress unused warning
            let where_clause = if conditions.is_empty() {
                String::new()
            } else {
                format!("WHERE {}", conditions.join(" AND "))
            };
            let sql = format!(
                "SELECT id, file_name, display_name, description, category, label_type, \
                 record_count, attack_count, benign_count, file_path, fetch_status, \
                 hf_uri, source_url, license \
                 FROM training_datasets {} ORDER BY category, id",
                where_clause
            );
            let mut q = sqlx::query(&sql);
            if let Some(c) = category   { q = q.bind(c); }
            if let Some(s) = status     { q = q.bind(s); }
            if let Some(l) = label_type { q = q.bind(l); }
            if let Some(li) = license   { q = q.bind(li); }
            let rows = q.fetch_all(pool).await?;
            Ok(rows.iter().map(|r| map_row!(r)).collect())
        }
    }
}
