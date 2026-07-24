// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! L0 normalization wrapper — delegates to parapet core `normalize_with_evidence`.

use parapet::normalize::normalize_with_evidence;

use crate::engine::verdict::NormalizationStats;

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Result of L0 normalization: cleaned text + evidence stats.
pub struct L0Result {
    pub normalized_text: String,
    pub stats: NormalizationStats,
}

/// Apply L0 normalization to raw input text.
///
/// Always runs — not configurable by the client.
pub fn normalize(text: &str) -> L0Result {
    let (normalized, evidence) = normalize_with_evidence(text);
    L0Result {
        stats: NormalizationStats {
            html_stripped: evidence.html_stripped,
            invisible_chars_removed: evidence.removed_invisible_count,
            confusable_replacements: evidence.confusable_replacement_count,
            input_chars: evidence.pre_char_len,
            output_chars: evidence.post_char_len,
        },
        normalized_text: normalized,
    }
}
