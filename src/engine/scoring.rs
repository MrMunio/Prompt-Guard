// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Shared n-gram extraction and SVM scoring helpers.
//!
//! These mirror the logic in `parapet/src/layers/l1.rs` but operate on
//! `HashMap<String, f64>` (dynamic weights) instead of `phf::Map` (compiled-in).
//! Used by both base and custom SVM model scorers.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// N-gram analyzer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NgramAnalyzer {
    /// `char_wb`: character n-grams with word-boundary space padding. Default.
    CharWb,
    /// `char`: raw character n-grams — good for adversarial/GCG suffixes.
    Char,
    /// `word`: whitespace-tokenized word n-grams — good for phrase patterns.
    Word,
}

// ---------------------------------------------------------------------------
// N-gram extraction
// ---------------------------------------------------------------------------

/// Extract unique n-grams from text using the specified analyzer and range.
pub fn extract_ngrams(text: &str, analyzer: NgramAnalyzer, range: (usize, usize)) -> Vec<String> {
    match analyzer {
        NgramAnalyzer::CharWb => extract_char_wb(text, range),
        NgramAnalyzer::Char   => extract_char(text, range),
        NgramAnalyzer::Word   => extract_word(text, range),
    }
}

fn extract_char_wb(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut seen = HashSet::new();
    let mut ngrams = Vec::new();
    for word in lower.split_whitespace() {
        let padded = format!(" {word} ");
        let chars: Vec<char> = padded.chars().collect();
        for n in min_n..=max_n {
            if chars.len() < n {
                let s: String = chars.iter().collect();
                if seen.insert(s.clone()) { ngrams.push(s); }
                continue;
            }
            for w in chars.windows(n) {
                let s: String = w.iter().collect();
                if seen.insert(s.clone()) { ngrams.push(s); }
            }
        }
    }
    ngrams
}

fn extract_char(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut seen = HashSet::new();
    let mut ngrams = Vec::new();
    for n in min_n..=max_n {
        if chars.len() < n { continue; }
        for w in chars.windows(n) {
            let s: String = w.iter().collect();
            if seen.insert(s.clone()) { ngrams.push(s); }
        }
    }
    ngrams
}

fn extract_word(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut seen = HashSet::new();
    let mut ngrams = Vec::new();
    for n in min_n..=max_n {
        if words.len() < n { continue; }
        for w in words.windows(n) {
            let s = w.join(" ");
            if seen.insert(s.clone()) { ngrams.push(s); }
        }
    }
    ngrams
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Dot-product of n-gram presence vector against dynamic weight map.
pub fn score_ngrams(ngrams: &[String], bias: f64, weights: &HashMap<String, f64>) -> f64 {
    let mut score = bias;
    for ngram in ngrams {
        if let Some(&w) = weights.get(ngram.as_str()) {
            score += w;
        }
    }
    score
}

// ---------------------------------------------------------------------------
// Sigmoid calibration
// ---------------------------------------------------------------------------

/// Sigmoid steepness — calibrated to match parapet L1 sigmoid (A=0.6, B=0.0).
const SIGMOID_A: f64 = 0.6;

/// Convert a raw SVM margin to a calibrated probability [0.0, 1.0].
pub fn calibrate_score(raw: f64) -> f32 {
    (1.0 / (1.0 + (-SIGMOID_A * raw).exp())) as f32
}

// ---------------------------------------------------------------------------
// Squash pass (de-obfuscation)
// ---------------------------------------------------------------------------

/// Strip non-alphanumeric chars and lowercase — mirrors parapet's `squash()`.
#[allow(dead_code)] // mirrors the Python squash() used in training pipeline
pub fn squash(text: &str) -> String {
    text.chars()
        .flat_map(|c| c.to_lowercase())
        .filter(|c| c.is_alphanumeric())
        .collect()
}
