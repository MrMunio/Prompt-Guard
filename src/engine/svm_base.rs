// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Base SVM model loader and runtime scorer.
//!
//! Each base model is a `.weights.json` file emitted by `train_base_models.py`:
//!
//! ```json
//! {
//!   "bias": -0.42,
//!   "weights": { " ign": 0.81, "ore ": 0.77, ... },
//!   "analyzer": "char_wb",
//!   "ngram_range": [3, 5]
//! }
//! ```
//!
//! At startup, all 9 base models are loaded into memory. Runtime scoring reuses
//! the same n-gram extraction and dot-product logic from parapet's L1 layer.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

use crate::engine::scoring::{calibrate_score, extract_ngrams, score_ngrams, NgramAnalyzer};

// ---------------------------------------------------------------------------
// The 9 canonical base model names
// ---------------------------------------------------------------------------

/// All valid base SVM category names. Order matters for display; "allrounder" is first.
pub const BASE_MODEL_NAMES: &[&str] = &[
    "allrounder",
    "instruction_override",
    "roleplay_jailbreak",
    "meta_probe",
    "exfiltration",
    "adversarial_suffix",
    "indirect_injection",
    "obfuscation",
    "constraint_bypass",
];

pub fn is_valid_base_category(name: &str) -> bool {
    BASE_MODEL_NAMES.contains(&name)
}

// ---------------------------------------------------------------------------
// Serialised weight file format
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WeightsFile {
    bias: f64,
    weights: HashMap<String, f64>,
    #[serde(default = "default_analyzer")]
    analyzer: String,
    #[serde(default = "default_ngram_range")]
    ngram_range: [usize; 2],
}

fn default_analyzer() -> String { "char_wb".to_string() }
fn default_ngram_range() -> [usize; 2] { [3, 5] }

// ---------------------------------------------------------------------------
// BaseSvmModel
// ---------------------------------------------------------------------------

/// A loaded base SVM model ready for runtime scoring.
pub struct BaseSvmModel {
    #[allow(dead_code)]
    pub name: String,
    pub bias: f64,
    pub weights: HashMap<String, f64>,
    pub analyzer: NgramAnalyzer,
    pub ngram_range: (usize, usize),
}

impl BaseSvmModel {
    /// Load from a `.weights.json` file.
    pub fn load(name: &str, path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read model file '{path:?}': {e}"))?;
        let wf: WeightsFile = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse model '{name}': {e}"))?;

        let analyzer = match wf.analyzer.as_str() {
            "char_wb" => NgramAnalyzer::CharWb,
            "char"    => NgramAnalyzer::Char,
            "word"    => NgramAnalyzer::Word,
            other     => anyhow::bail!("Unknown analyzer '{other}' in model '{name}'"),
        };

        Ok(Self {
            name: name.to_string(),
            bias: wf.bias,
            weights: wf.weights,
            analyzer,
            ngram_range: (wf.ngram_range[0], wf.ngram_range[1]),
        })
    }

    /// Score text: returns raw SVM margin (unbounded).
    pub fn score_raw(&self, text: &str) -> f64 {
        let ngrams = extract_ngrams(text, self.analyzer, self.ngram_range);
        score_ngrams(&ngrams, self.bias, &self.weights)
    }

    /// Score text: returns calibrated probability [0.0, 1.0].
    pub fn score(&self, text: &str) -> f32 {
        calibrate_score(self.score_raw(text))
    }
}

// ---------------------------------------------------------------------------
// Registry loaded at startup
// ---------------------------------------------------------------------------

/// All 9 base models loaded into memory.
pub struct BaseModelRegistry {
    models: HashMap<String, Arc<BaseSvmModel>>,
}

impl BaseModelRegistry {
    /// Load all base models from `models_dir/base/`.
    pub fn load(models_dir: &str) -> anyhow::Result<Self> {
        let base_dir = Path::new(models_dir).join("base");
        let mut models = HashMap::new();

        for name in BASE_MODEL_NAMES {
            let path = base_dir.join(format!("{name}.weights.json"));
            if !path.exists() {
                anyhow::bail!(
                    "Base model file not found: {path:?}. \
                     Run 'python scripts/train_base_models.py' to generate base models."
                );
            }
            let model = BaseSvmModel::load(name, &path)?;
            tracing::info!(model = name, "base SVM model loaded");
            models.insert(name.to_string(), Arc::new(model));
        }

        Ok(Self { models })
    }

    /// Get a model by name. Returns None if not loaded.
    pub fn get(&self, name: &str) -> Option<Arc<BaseSvmModel>> {
        self.models.get(name).cloned()
    }

    /// Returns all model names.
    #[allow(dead_code)]
    pub fn all_names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}
