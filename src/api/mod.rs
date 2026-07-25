// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Router assembly — wires all API routes with auth middleware.

pub mod datasets;
pub mod detect;
pub mod health;
pub mod models;
pub mod patterns;
pub mod train;

use std::sync::Arc;

use axum::{
    middleware,
    routing::{delete, get, post, put},
    Extension, Router,
};
use tower_http::cors::CorsLayer;

use crate::auth::{require_api_key, ApiKeyConfig};
use crate::db::DbPool;
use crate::engine::EngineState;

// ---------------------------------------------------------------------------
// App state — shared across all handlers
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<DbPool>,
    pub engine: Arc<EngineState>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn build_router(state: AppState, api_key: String) -> Router {
    // Health is public — no auth.
    let public = Router::new().route("/v1/health", get(health::health_handler));

    // All other routes require X-API-Key.
    let protected = Router::new()
        // Detect
        .route("/v1/detect", post(detect::detect_handler))
        // Pattern groups
        .route("/v1/patterns", post(patterns::create_pattern_group))
        .route("/v1/patterns", get(patterns::list_pattern_groups))
        .route("/v1/patterns/{id}", get(patterns::get_pattern_group))
        .route("/v1/patterns/{id}", put(patterns::update_pattern_group))
        .route("/v1/patterns/{id}", delete(patterns::delete_pattern_group))
        .route("/v1/patterns/{id}/entries", post(patterns::add_pattern_entries))
        .route("/v1/patterns/{id}/entries/{entry_id}", delete(patterns::delete_pattern_entry))
        // Custom models
        .route("/v1/models", post(models::create_model))
        .route("/v1/models", get(models::list_models))
        .route("/v1/models/{id}", get(models::get_model))
        .route("/v1/models/{id}", delete(models::delete_model))
        // Training
        .route("/v1/models/{id}/train", post(train::train_handler))
        .route("/v1/models/{id}/training-status", get(train::training_status_handler))
        // Dataset catalog
        .route("/v1/datasets", get(datasets::list_datasets_handler))
        .route("/v1/datasets/{id}/fetch", post(datasets::fetch_dataset_handler))
        // Auth middleware on all protected routes
        .layer(middleware::from_fn(require_api_key))
        .layer(Extension(ApiKeyConfig { key: api_key }));

    Router::new()
        .merge(public)
        .merge(protected)
        .layer(CorsLayer::permissive())
        .with_state(state)
}
