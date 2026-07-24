// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Custom regex pattern scanner — loads patterns from the database and scans text.

use regex::Regex;
use std::collections::HashMap;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Pattern record (as returned from DB query)
// ---------------------------------------------------------------------------

/// Represents a pattern_entries row joined with its group metadata.
#[derive(Debug, Clone)]
pub struct PatternGroupRecord {
    pub group_id: String,
    pub group_name: String,
    pub description: Option<String>,
    pub category: Option<String>,
    /// All compiled regex patterns belonging to this group.
    pub patterns: Vec<(String, Regex)>, // (raw_pattern_string, compiled)
}

// ---------------------------------------------------------------------------
// Compiled pattern cache
// ---------------------------------------------------------------------------

/// Caches compiled regex sets by group ID. Invalidated when patterns are added/removed.
pub struct CustomRegexCache {
    cache: RwLock<HashMap<String, PatternGroupRecord>>,
}

impl CustomRegexCache {
    pub fn new() -> Self {
        Self { cache: RwLock::new(HashMap::new()) }
    }

    /// Store a compiled pattern group.
    pub async fn insert(&self, record: PatternGroupRecord) {
        self.cache.write().await.insert(record.group_id.clone(), record);
    }

    /// Retrieve a cached pattern group.
    pub async fn get(&self, group_id: &str) -> Option<PatternGroupRecord> {
        self.cache.read().await.get(group_id).cloned()
    }

    /// Evict a group (e.g. after pattern add/delete).
    pub async fn evict(&self, group_id: &str) {
        self.cache.write().await.remove(group_id);
    }
}

impl Default for CustomRegexCache {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------

/// Scan text against a set of custom pattern group records.
/// Returns `(score, matched_pattern_strings)`.
pub fn scan_custom_patterns(text: &str, group: &PatternGroupRecord) -> (f32, Vec<String>) {
    let mut matched = Vec::new();
    for (raw, compiled) in &group.patterns {
        if compiled.is_match(text) {
            matched.push(raw.clone());
        }
    }
    let score = if matched.is_empty() { 0.0 } else { 1.0 };
    (score, matched)
}

/// Try to compile a regex pattern string.
/// Strings without any regex special characters/anchors are treated as plain text descriptions.
pub fn compile_pattern(pattern: &str) -> Result<Regex, String> {
    let has_regex_meta = pattern.contains('(') || pattern.contains('[') || 
                         pattern.contains('^') || pattern.contains('$') || 
                         pattern.contains('*') || pattern.contains('+') || 
                         pattern.contains('?') || pattern.contains('\\') ||
                         pattern.contains('|');
    if !has_regex_meta {
        return Err(format!("Plain text description without regex syntax: '{pattern}'"));
    }
    Regex::new(pattern).map_err(|e| format!("Invalid regex '{pattern}': {e}"))
}
