// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! API key authentication middleware.
//!
//! All routes except `GET /v1/health` require the `X-API-Key` header to match
//! the `API_KEY` environment variable. Comparison is constant-time to prevent
//! timing attacks.

use axum::{
    extract::Request,
    http::HeaderMap,
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// axum middleware that enforces API key authentication.
///
/// Install with `axum::middleware::from_fn_with_state` on all protected routes.
pub async fn require_api_key(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // The configured API key is passed via extension set in the router.
    // We read it from the request extensions.
    let expected = request
        .extensions()
        .get::<ApiKeyConfig>()
        .map(|c| c.key.as_bytes())
        .unwrap_or(b"");

    let provided = headers
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // Constant-time comparison — prevents timing side-channels.
    let matches: bool = expected.ct_eq(provided.as_bytes()).into();
    if !matches {
        return Err(ApiError::Unauthorized);
    }

    Ok(next.run(request).await)
}

// ---------------------------------------------------------------------------
// Extension type carrying the expected key into the middleware
// ---------------------------------------------------------------------------

/// Holds the expected API key, injected as an axum Extension at router build time.
#[derive(Clone)]
pub struct ApiKeyConfig {
    pub key: String,
}
