// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Pattern group CRUD API handlers.
//!
//! POST   /v1/patterns              — create group (LLM regex gen if needed)
//! GET    /v1/patterns              — list groups
//! GET    /v1/patterns/:id          — get group + entries
//! PUT    /v1/patterns/:id          — update name/description/category
//! DELETE /v1/patterns/:id          — delete group (cascades entries)
//! POST   /v1/patterns/:id/entries  — add more entries
//! DELETE /v1/patterns/:id/entries/:entry_id — remove one entry
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
use crate::engine::regex_custom::compile_pattern;
use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreatePatternGroupRequest {
    pub id: Option<String>,
    pub name: String,
    pub description: Option<String>,
    pub category: Option<String>,
    /// One or more strings: plain-text intent or actual regex patterns.
    pub input: Vec<String>,
}

#[derive(Deserialize)]
pub struct UpdatePatternGroupRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
}

#[derive(Deserialize)]
pub struct AddPatternEntriesRequest {
    pub input: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct PatternGroupResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: Option<String>,
    pub entries: Vec<PatternEntryResponse>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct PatternEntryResponse {
    pub id: String,
    pub pattern: String,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/patterns — create a new pattern group with entries.
pub async fn create_pattern_group(
    State(state): State<AppState>,
    Json(req): Json<CreatePatternGroupRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::bad("'name' must not be empty"));
    }
    if req.input.is_empty() {
        return Err(ApiError::bad("'input' must contain at least one pattern or description"));
    }

    // Determine which inputs are regex and which need LLM generation.
    let mut patterns: Vec<String> = Vec::new();
    let mut needs_llm: Vec<String> = Vec::new();

    for input in &req.input {
        if compile_pattern(input).is_ok() {
            patterns.push(input.clone());
        } else {
            needs_llm.push(input.clone());
        }
    }

    // Generate regex for plain-text descriptions via LLM (if configured).
    for desc in &needs_llm {
        match generate_regex_via_llm(desc, &state).await {
            Ok(generated) => patterns.extend(generated),
            Err(e) => {
                tracing::warn!(description = desc, error = %e, "LLM regex generation skipped — storing description as literal pattern");
                patterns.push(regex::escape(desc));
            }
        }
    }

    if patterns.is_empty() {
        return Err(ApiError::bad("Could not produce any valid regex patterns. Provide valid regex input."));
    }

    let group_id = req.id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let now = Utc::now().to_rfc3339();

    // Insert group.
    match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "INSERT INTO pattern_groups (id, name, description, category, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?)"
            ).bind(&group_id).bind(&req.name).bind(&req.description).bind(&req.category)
             .bind(&now).bind(&now).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO pattern_groups (id, name, description, category, created_at, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6)"
            ).bind(&group_id).bind(&req.name).bind(&req.description).bind(&req.category)
             .bind(&now).bind(&now).execute(pool).await?;
        }
    }

    // Insert each pattern entry.
    for pattern in &patterns {
        insert_pattern_entry(&group_id, pattern, &state.db, &now).await?;
    }

    // Invalidate custom regex cache for this group.
    state.engine.regex_custom_cache.evict(&group_id).await;

    let resp = load_pattern_group_response(&group_id, &state.db).await?;
    Ok((axum::http::StatusCode::CREATED, Json(resp)))
}

/// GET /v1/patterns
pub async fn list_pattern_groups(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let group_ids: Vec<String> = match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("SELECT id FROM pattern_groups ORDER BY created_at DESC")
                .fetch_all(pool).await?
                .iter().map(|r| r.get::<String, _>("id")).collect()
        }
        DbPool::Postgres(pool) => {
            sqlx::query("SELECT id FROM pattern_groups ORDER BY created_at DESC")
                .fetch_all(pool).await?
                .iter().map(|r| r.get::<String, _>("id")).collect()
        }
    };

    let mut groups = Vec::new();
    for id in &group_ids {
        groups.push(load_pattern_group_response(id, &state.db).await?);
    }
    Ok(Json(serde_json::json!({ "pattern_groups": groups })))
}

/// GET /v1/patterns/:id
pub async fn get_pattern_group(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let resp = load_pattern_group_response(&id, &state.db).await?;
    Ok(Json(resp))
}

/// PUT /v1/patterns/:id
pub async fn update_pattern_group(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdatePatternGroupRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Check exists.
    let _ = load_pattern_group_response(&id, &state.db).await?;
    let now = Utc::now().to_rfc3339();

    match &*state.db {
        DbPool::Sqlite(pool) => {
            if let Some(n) = &req.name {
                sqlx::query("UPDATE pattern_groups SET name=?, updated_at=? WHERE id=?")
                    .bind(n).bind(&now).bind(&id).execute(pool).await?;
            }
            if let Some(d) = &req.description {
                sqlx::query("UPDATE pattern_groups SET description=?, updated_at=? WHERE id=?")
                    .bind(d).bind(&now).bind(&id).execute(pool).await?;
            }
            if let Some(c) = &req.category {
                sqlx::query("UPDATE pattern_groups SET category=?, updated_at=? WHERE id=?")
                    .bind(c).bind(&now).bind(&id).execute(pool).await?;
            }
        }
        DbPool::Postgres(pool) => {
            if let Some(n) = &req.name {
                sqlx::query("UPDATE pattern_groups SET name=$1, updated_at=$2 WHERE id=$3")
                    .bind(n).bind(&now).bind(&id).execute(pool).await?;
            }
            if let Some(d) = &req.description {
                sqlx::query("UPDATE pattern_groups SET description=$1, updated_at=$2 WHERE id=$3")
                    .bind(d).bind(&now).bind(&id).execute(pool).await?;
            }
            if let Some(c) = &req.category {
                sqlx::query("UPDATE pattern_groups SET category=$1, updated_at=$2 WHERE id=$3")
                    .bind(c).bind(&now).bind(&id).execute(pool).await?;
            }
        }
    }

    state.engine.regex_custom_cache.evict(&id).await;
    let resp = load_pattern_group_response(&id, &state.db).await?;
    Ok(Json(resp))
}

/// DELETE /v1/patterns/:id
pub async fn delete_pattern_group(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = load_pattern_group_response(&id, &state.db).await?; // 404 check

    match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("DELETE FROM pattern_groups WHERE id=?")
                .bind(&id).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query("DELETE FROM pattern_groups WHERE id=$1")
                .bind(&id).execute(pool).await?;
        }
    }

    state.engine.regex_custom_cache.evict(&id).await;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

/// POST /v1/patterns/:id/entries
pub async fn add_pattern_entries(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<AddPatternEntriesRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = load_pattern_group_response(&id, &state.db).await?; // 404 check

    let mut patterns: Vec<String> = Vec::new();
    for input in &req.input {
        if compile_pattern(input).is_ok() {
            patterns.push(input.clone());
        } else {
            match generate_regex_via_llm(input, &state).await {
                Ok(generated) => patterns.extend(generated),
                Err(e) => {
                    tracing::warn!(description = input, error = %e, "LLM regex generation skipped — storing description as literal pattern");
                    patterns.push(regex::escape(input));
                }
            }
        }
    }

    let now = Utc::now().to_rfc3339();
    for pattern in &patterns {
        insert_pattern_entry(&id, pattern, &state.db, &now).await?;
    }

    state.engine.regex_custom_cache.evict(&id).await;
    let resp = load_pattern_group_response(&id, &state.db).await?;
    Ok(Json(resp))
}

/// DELETE /v1/patterns/:id/entries/:entry_id
pub async fn delete_pattern_entry(
    State(state): State<AppState>,
    Path((group_id, entry_id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    match &*state.db {
        DbPool::Sqlite(pool) => {
            sqlx::query("DELETE FROM pattern_entries WHERE id=? AND group_id=?")
                .bind(&entry_id).bind(&group_id).execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query("DELETE FROM pattern_entries WHERE id=$1 AND group_id=$2")
                .bind(&entry_id).bind(&group_id).execute(pool).await?;
        }
    }
    state.engine.regex_custom_cache.evict(&group_id).await;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// DB helpers
// ---------------------------------------------------------------------------

async fn insert_pattern_entry(
    group_id: &str,
    pattern: &str,
    db: &DbPool,
    now: &str,
) -> Result<String, ApiError> {
    let entry_id = Uuid::new_v4().to_string();
    match db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "INSERT INTO pattern_entries (id, group_id, raw_input, pattern, source, created_at) VALUES (?, ?, ?, ?, 'user_regex', ?)"
            ).bind(&entry_id).bind(group_id).bind(pattern).bind(pattern).bind(now)
             .execute(pool).await?;
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO pattern_entries (id, group_id, raw_input, pattern, source, created_at) VALUES ($1, $2, $3, $4, 'user_regex', $5)"
            ).bind(&entry_id).bind(group_id).bind(pattern).bind(pattern).bind(now)
             .execute(pool).await?;
        }
    }
    Ok(entry_id)
}

pub async fn load_pattern_group_response(
    group_id: &str,
    db: &DbPool,
) -> Result<PatternGroupResponse, ApiError> {
    // Fetch group row.
    let (name, description, category, created_at, updated_at): (String, Option<String>, Option<String>, String, String) = match db {
        DbPool::Sqlite(pool) => {
            let r = sqlx::query(
                "SELECT name, description, category, created_at, updated_at FROM pattern_groups WHERE id = ?"
            ).bind(group_id).fetch_optional(pool).await?
             .ok_or_else(|| ApiError::NotFound(format!("Pattern group '{group_id}' not found")))?;
            (r.get("name"), r.get("description"), r.get("category"),
             r.get("created_at"), r.get("updated_at"))
        }
        DbPool::Postgres(pool) => {
            let r = sqlx::query(
                "SELECT name, description, category, created_at, updated_at FROM pattern_groups WHERE id = $1"
            ).bind(group_id).fetch_optional(pool).await?
             .ok_or_else(|| ApiError::NotFound(format!("Pattern group '{group_id}' not found")))?;
            (r.get("name"), r.get("description"), r.get("category"),
             r.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
             r.get::<chrono::DateTime<chrono::Utc>, _>("updated_at").to_rfc3339())
        }
    };

    // Fetch entries.
    let entries: Vec<PatternEntryResponse> = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "SELECT id, pattern, created_at FROM pattern_entries WHERE group_id = ? ORDER BY created_at"
            ).bind(group_id).fetch_all(pool).await?
             .iter().map(|r| PatternEntryResponse {
                 id: r.get("id"),
                 pattern: r.get("pattern"),
                 created_at: r.get("created_at"),
             }).collect()
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "SELECT id, pattern, created_at FROM pattern_entries WHERE group_id = $1 ORDER BY created_at"
            ).bind(group_id).fetch_all(pool).await?
             .iter().map(|r| PatternEntryResponse {
                 id: r.get("id"),
                 pattern: r.get("pattern"),
                 created_at: r.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
             }).collect()
        }
    };

    Ok(PatternGroupResponse {
        id: group_id.to_string(),
        name,
        description,
        category,
        entries,
        created_at,
        updated_at,
    })
}

// ---------------------------------------------------------------------------
// LLM regex generation
// ---------------------------------------------------------------------------

async fn generate_regex_via_llm(description: &str, state: &AppState) -> Result<Vec<String>, ApiError> {
    let output = tokio::process::Command::new(&state.engine.python_executable)
        .arg("scripts/generate_regex.py")
        .arg("--description").arg(description)
        .arg("--base-url").arg(&state.engine.llm_base_url)
        .arg("--model").arg(&state.engine.llm_model)
        .arg("--api-key").arg(&state.engine.llm_api_key)
        .output().await
        .map_err(|e| ApiError::Internal(format!("Failed to run generate_regex.py: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ApiError::Internal(format!("generate_regex.py failed: {stderr}")));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| ApiError::Internal(format!("generate_regex.py produced invalid JSON: {e}")))?;

    let patterns: Vec<String> = parsed["patterns"]
        .as_array()
        .ok_or_else(|| ApiError::Internal("generate_regex.py: expected 'patterns' array".to_string()))?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();

    Ok(patterns)
}
