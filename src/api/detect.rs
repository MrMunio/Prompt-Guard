// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/detect` — the main detection endpoint.


use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::api::AppState;
use crate::engine::l0;
use crate::engine::verdict::{DetectResponse, VerdictValue};
use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct DetectRequest {
    /// Combined text to scan (user query + document content).
    pub text: String,

    /// Guardrail selection — which checks to run.
    pub guardrails: GuardrailConfig,
}

#[derive(Debug, Deserialize)]
pub struct GuardrailConfig {
    /// Base SVM categories to evaluate. "all" → allrounder model.
    #[serde(default)]
    pub svm_base: SvmBaseInput,

    /// Custom model IDs to evaluate. "all" → all registered custom models with a trained weight file.
    #[serde(default)]
    pub svm_custom: SvmCustomInput,

    /// Built-in regex categories to evaluate. "all" → every base category.
    #[serde(default)]
    pub regex_base: RegexBaseInput,

    /// Custom pattern group IDs to evaluate. "all" → every registered group.
    #[serde(default)]
    pub regex_custom: RegexCustomInput,
}

// ── Serde: each field accepts either a string "all" or an array of strings ──

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
pub enum SvmBaseInput {
    All(AllTag),
    Categories(Vec<String>),
    #[default]
    Empty,
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
pub enum SvmCustomInput {
    All(AllTag),
    Ids(Vec<String>),
    #[default]
    Empty,
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
pub enum RegexBaseInput {
    All(AllTag),
    Categories(Vec<String>),
    #[default]
    Empty,
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
pub enum RegexCustomInput {
    All(AllTag),
    Ids(Vec<String>),
    #[default]
    Empty,
}

/// Helper type that deserialises only the literal string "all".
#[derive(Debug, Deserialize)]
#[serde(try_from = "String")]
pub struct AllTag;

impl TryFrom<String> for AllTag {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "all" { Ok(AllTag) } else { Err(format!("expected \"all\", got \"{s}\"")) }
    }
}

// ── Resolved selector types (used internally by engine) ──────────────────────

#[derive(Debug)]
pub enum SvmBaseSelector  { All, Categories(Vec<String>), None }
#[derive(Debug)]
pub enum SvmCustomSelector { All, Ids(Vec<String>), None }
#[derive(Debug)]
pub enum RegexBaseSelector { All, Categories(Vec<String>), None }
#[derive(Debug)]
pub enum RegexCustomSelector { All, Ids(Vec<String>), None }

/// Parsed + validated guardrail selector passed to the engine.
#[derive(Debug)]
pub struct GuardrailSelector {
    pub svm_base: SvmBaseSelector,
    pub svm_custom: SvmCustomSelector,
    pub regex_base: RegexBaseSelector,
    pub regex_custom: RegexCustomSelector,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub async fn detect_handler(
    State(state): State<AppState>,
    Json(payload): Json<DetectRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // ── Validate text ────────────────────────────────────────────────────────
    if payload.text.trim().is_empty() {
        return Err(ApiError::bad("'text' must not be empty"));
    }
    // Max text length comes from AppState (set from env at startup).
    let max_chars = state.engine.max_text_chars;
    if payload.text.len() > max_chars {
        return Err(ApiError::bad(format!(
            "'text' exceeds maximum length of {max_chars} characters"
        )));
    }

    // ── Validate guardrail selector ──────────────────────────────────────────
    let selector = validate_selector(&payload.guardrails, &state)?;

    // ── L0 — always runs ─────────────────────────────────────────────────────
    let l0_result = l0::normalize(&payload.text);
    let normalized_text = l0_result.normalized_text;

    // ── Run selected guardrails ──────────────────────────────────────────────
    let (results, composite_score) = state
        .engine
        .run(&normalized_text, &selector, &state.db)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // ── Aggregate verdict ────────────────────────────────────────────────────
    let verdict = if results.iter().any(|r| r.verdict == VerdictValue::Block) {
        VerdictValue::Block
    } else {
        VerdictValue::Allow
    };

    Ok(Json(DetectResponse {
        verdict,
        composite_score,
        normalization: l0_result.stats,
        results,
    }).into_response())
}

// ---------------------------------------------------------------------------
// Selector validation
// ---------------------------------------------------------------------------

use crate::engine::svm_base::is_valid_base_category;

fn validate_selector(
    cfg: &GuardrailConfig,
    _state: &AppState,
) -> Result<GuardrailSelector, ApiError> {
    // At least one guardrail must be selected.
    let all_empty = matches!(cfg.svm_base, SvmBaseInput::Empty)
        && matches!(cfg.svm_custom, SvmCustomInput::Empty)
        && matches!(cfg.regex_base, RegexBaseInput::Empty)
        && matches!(cfg.regex_custom, RegexCustomInput::Empty);

    if all_empty {
        return Err(ApiError::bad(
            "At least one guardrail selector (svm_base, svm_custom, regex_base, regex_custom) must be provided",
        ));
    }

    // Validate svm_base category names.
    let svm_base = match &cfg.svm_base {
        SvmBaseInput::All(_) => SvmBaseSelector::All,
        SvmBaseInput::Categories(cats) => {
            let invalid: Vec<_> = cats.iter().filter(|c| !is_valid_base_category(c.as_str())).collect();
            if !invalid.is_empty() {
                return Err(ApiError::bad_fields(
                    "Invalid svm_base category names",
                    serde_json::json!({ "svm_base": invalid }),
                ));
            }
            SvmBaseSelector::Categories(cats.clone())
        }
        SvmBaseInput::Empty => SvmBaseSelector::None,
    };

    let svm_custom = match &cfg.svm_custom {
        SvmCustomInput::All(_) => SvmCustomSelector::All,
        SvmCustomInput::Ids(ids) => SvmCustomSelector::Ids(ids.clone()),
        SvmCustomInput::Empty => SvmCustomSelector::None,
    };

    let regex_base = match &cfg.regex_base {
        RegexBaseInput::All(_) => RegexBaseSelector::All,
        RegexBaseInput::Categories(cats) => {
            let invalid: Vec<_> = cats.iter().filter(|c| !is_valid_base_category(c.as_str()) || c.as_str() == "allrounder").collect();
            if !invalid.is_empty() {
                return Err(ApiError::bad_fields(
                    "Invalid regex_base category names",
                    serde_json::json!({ "regex_base": invalid }),
                ));
            }
            RegexBaseSelector::Categories(cats.clone())
        }
        RegexBaseInput::Empty => RegexBaseSelector::None,
    };

    let regex_custom = match &cfg.regex_custom {
        RegexCustomInput::All(_) => RegexCustomSelector::All,
        RegexCustomInput::Ids(ids) => RegexCustomSelector::Ids(ids.clone()),
        RegexCustomInput::Empty => RegexCustomSelector::None,
    };

    Ok(GuardrailSelector { svm_base, svm_custom, regex_base, regex_custom })
}
