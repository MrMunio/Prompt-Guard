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
//! At startup, base models are loaded into memory. Alias mapping is supported so that
//! categories without dedicated weight files share the single `allrounder` model instance,
//! saving RAM and avoiding duplicate SVM matrix operations at runtime.

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
    "allrounder_legacy",
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

/// Base models loaded into memory (with alias fallback to allrounder).
pub struct BaseModelRegistry {
    models: HashMap<String, Arc<BaseSvmModel>>,
}

impl BaseModelRegistry {
    /// Load base models from `models_dir/base/`.
    /// If specific category `.weights.json` files exist, they are loaded.
    /// Otherwise, categories share the single `allrounder` model instance.
    pub fn load(models_dir: &str) -> anyhow::Result<Self> {
        let base_dir = Path::new(models_dir).join("base");
        let mut models = HashMap::new();

        // 1. Ensure at least allrounder.weights.json exists
        let allrounder_path = base_dir.join("allrounder.weights.json");
        if !allrounder_path.exists() {
            anyhow::bail!(
                "Base model file not found: {allrounder_path:?}. \
                 Run 'python scripts/train_base_models.py' to generate base models."
            );
        }

        let allrounder_model = Arc::new(BaseSvmModel::load("allrounder", &allrounder_path)?);
        tracing::info!("base SVM model loaded: allrounder (shared generalist)");
        models.insert("allrounder".to_string(), allrounder_model.clone());

        // 2. Load allrounder_legacy (L1 sparse baseline) if present, else alias to allrounder
        let legacy_path = base_dir.join("allrounder_legacy.weights.json");
        if legacy_path.exists() {
            let legacy_model = Arc::new(BaseSvmModel::load("allrounder_legacy", &legacy_path)?);
            tracing::info!("base SVM model loaded: allrounder_legacy (L1 sparse baseline)");
            models.insert("allrounder_legacy".to_string(), legacy_model);
        } else {
            tracing::info!(category = "allrounder_legacy", target = "allrounder", "aliasing allrounder_legacy to allrounder (weights file not found)");
            models.insert("allrounder_legacy".to_string(), allrounder_model.clone());
        }

        // 3. Load remaining category-specific models or alias to allrounder
        for &name in BASE_MODEL_NAMES {
            if name == "allrounder" || name == "allrounder_legacy" {
                continue;
            }
            let path = base_dir.join(format!("{name}.weights.json"));
            if path.exists() {
                let model = BaseSvmModel::load(name, &path)?;
                tracing::info!(model = name, "base SVM model loaded (dedicated)");
                models.insert(name.to_string(), Arc::new(model));
            } else {
                tracing::info!(category = name, target = "allrounder", "aliasing category to allrounder");
                models.insert(name.to_string(), allrounder_model.clone());
            }
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
