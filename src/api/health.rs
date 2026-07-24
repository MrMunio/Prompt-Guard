// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `GET /v1/health` — liveness probe. No authentication required.

use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

pub async fn health_handler() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "service": "parapet-guardrail"
    }))
}
