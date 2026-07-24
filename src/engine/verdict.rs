// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Detection result types — the structured JSON response for POST /v1/detect.

use serde::Serialize;

// ---------------------------------------------------------------------------
// Top-level detect response
// ---------------------------------------------------------------------------

/// Full detection response returned to the caller.
#[derive(Debug, Clone, Serialize)]
pub struct DetectResponse {
    /// Aggregate verdict across all selected guardrails: "block" | "allow".
    pub verdict: VerdictValue,

    /// Highest composite score across all active guardrails [0.0, 1.0].
    pub composite_score: f32,

    /// L0 normalization stats (always present — L0 always runs).
    pub normalization: NormalizationStats,

    /// Per-guardrail attribution detail.
    pub results: Vec<GuardrailResult>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum VerdictValue {
    Allow,
    Block,
}

// ---------------------------------------------------------------------------
// Per-guardrail result
// ---------------------------------------------------------------------------

/// Attribution record from a single guardrail check.
#[derive(Debug, Clone, Serialize)]
pub struct GuardrailResult {
    /// Stable identifier: category name for base, UUID for custom.
    pub guardrail_id: String,

    /// "svm" or "regex".
    pub guardrail_type: GuardrailType,

    /// "base" (built-in) or "custom" (client-registered).
    pub source: GuardrailSource,

    /// Human-readable name of this guardrail.
    pub name: String,

    /// Attack category label if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    /// Short description of what this guardrail detects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Verdict for this specific guardrail.
    pub verdict: VerdictValue,

    /// Score [0.0, 1.0] — calibrated SVM probability or 1.0/0.0 for regex match/no-match.
    pub score: f32,

    /// Regex patterns that matched (only present for regex guardrails on block).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_patterns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardrailType {
    Svm,
    Regex,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardrailSource {
    Base,
    Custom,
}

// ---------------------------------------------------------------------------
// L0 normalization stats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct NormalizationStats {
    pub html_stripped: bool,
    pub invisible_chars_removed: usize,
    pub confusable_replacements: usize,
    pub input_chars: usize,
    pub output_chars: usize,
}
