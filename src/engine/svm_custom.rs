// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Custom SVM model loader — loads client-trained `.weights.json` files on demand.
//! Uses an in-memory LRU-style cache (simple HashMap for MVP; can be upgraded to LRU later).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::engine::scoring::{calibrate_score, extract_ngrams, score_ngrams, NgramAnalyzer};

// ---------------------------------------------------------------------------
// CustomSvmModel — same structure as BaseSvmModel but loaded from custom path
// ---------------------------------------------------------------------------

use serde::Deserialize;

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

pub struct CustomSvmModel {
    #[allow(dead_code)]
    pub id: String,
    pub bias: f64,
    pub weights: HashMap<String, f64>,
    pub analyzer: NgramAnalyzer,
    pub ngram_range: (usize, usize),
}

impl CustomSvmModel {
    pub fn load(id: &str, path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read custom model '{id}' at {path:?}: {e}"))?;
        let wf: WeightsFile = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Cannot parse custom model '{id}': {e}"))?;
        let analyzer = match wf.analyzer.as_str() {
            "char_wb" => NgramAnalyzer::CharWb,
            "char"    => NgramAnalyzer::Char,
            "word"    => NgramAnalyzer::Word,
            other     => anyhow::bail!("Unknown analyzer '{other}' in custom model '{id}'"),
        };
        Ok(Self {
            id: id.to_string(),
            bias: wf.bias,
            weights: wf.weights,
            analyzer,
            ngram_range: (wf.ngram_range[0], wf.ngram_range[1]),
        })
    }

    /// Score text: returns calibrated probability [0.0, 1.0].
    pub fn score(&self, text: &str) -> f32 {
        let ngrams = extract_ngrams(text, self.analyzer, self.ngram_range);
        calibrate_score(score_ngrams(&ngrams, self.bias, &self.weights))
    }
}

// ---------------------------------------------------------------------------
// Registry (in-memory cache)
// ---------------------------------------------------------------------------

/// Caches loaded custom models by ID to avoid disk reads on every request.
pub struct CustomModelCache {
    cache: RwLock<HashMap<String, Arc<CustomSvmModel>>>,
}

impl CustomModelCache {
    pub fn new() -> Self {
        Self { cache: RwLock::new(HashMap::new()) }
    }

    /// Get a model from cache or load from disk.
    pub async fn get(&self, id: &str, path: &str) -> anyhow::Result<Arc<CustomSvmModel>> {
        // Fast path: check cache under read lock.
        {
            let cache = self.cache.read().await;
            if let Some(m) = cache.get(id) {
                return Ok(m.clone());
            }
        }
        // Slow path: load from disk, insert under write lock.
        let model = CustomSvmModel::load(id, Path::new(path))?;
        let model = Arc::new(model);
        self.cache.write().await.insert(id.to_string(), model.clone());
        Ok(model)
    }

    /// Evict a model from cache (e.g. after retraining).
    pub async fn evict(&self, id: &str) {
        self.cache.write().await.remove(id);
    }
}

impl Default for CustomModelCache {
    fn default() -> Self { Self::new() }
}
