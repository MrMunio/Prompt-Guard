// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Unified error types for the guardrail API.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

// ---------------------------------------------------------------------------
// API error enum
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ApiError {
    /// 400 — malformed or semantically invalid request.
    #[error("bad request: {message}")]
    BadRequest {
        message: String,
        /// Optional per-field error detail map.
        fields: Option<serde_json::Value>,
    },

    /// 401 — missing or invalid API key.
    #[error("unauthorized")]
    Unauthorized,

    /// 404 — referenced resource (model/pattern) not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// 409 — a resource with this ID or name already exists.
    #[error("conflict: {0}")]
    Conflict(String),

    /// 422 — request structurally valid but business logic rejected it.
    #[error("unprocessable: {0}")]
    Unprocessable(String),

    /// 500 — unexpected internal failure.
    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    pub fn bad(msg: impl Into<String>) -> Self {
        Self::BadRequest { message: msg.into(), fields: None }
    }

    pub fn bad_fields(msg: impl Into<String>, fields: serde_json::Value) -> Self {
        Self::BadRequest { message: msg.into(), fields: Some(fields) }
    }
}

// ---------------------------------------------------------------------------
// axum response conversion
// ---------------------------------------------------------------------------

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message, fields) = match &self {
            ApiError::BadRequest { message, fields } => {
                (StatusCode::BAD_REQUEST, "bad_request", message.clone(), fields.clone())
            }
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "Invalid or missing X-API-Key header".to_string(),
                None,
            ),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "not_found", m.clone(), None),
            ApiError::Conflict(m) => (StatusCode::CONFLICT, "conflict", m.clone(), None),
            ApiError::Unprocessable(m) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "unprocessable",
                m.clone(),
                None,
            ),
            ApiError::Internal(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                m.clone(),
                None,
            ),
        };

        let mut body = json!({ "error": code, "message": message });
        if let Some(f) = fields {
            body["fields"] = f;
        }

        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// From conversions for common error sources
// ---------------------------------------------------------------------------

impl From<sqlx::Error> for ApiError {
    fn from(e: sqlx::Error) -> Self {
        tracing::error!(error = %e, "database error");
        ApiError::Internal(format!("database error: {e}"))
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        tracing::error!(error = %e, "internal error");
        ApiError::Internal(e.to_string())
    }
}
