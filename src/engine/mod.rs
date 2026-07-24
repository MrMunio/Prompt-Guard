// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! GuardrailEngine — orchestrates L0 → selected SVMs → selected regex → results.

pub mod l0;
pub mod regex_base;
pub mod regex_custom;
pub mod scoring;
pub mod svm_base;
pub mod svm_custom;
pub mod verdict;

use std::sync::Arc;

use crate::api::detect::{
    GuardrailSelector, RegexBaseSelector, RegexCustomSelector, SvmBaseSelector, SvmCustomSelector,
};
use crate::db::DbPool;
use crate::engine::regex_base::BaseRegexScanner;
use crate::engine::regex_custom::{
    compile_pattern, scan_custom_patterns, CustomRegexCache, PatternGroupRecord,
};
use crate::engine::svm_base::BaseModelRegistry;
use crate::engine::svm_custom::CustomModelCache;
use crate::engine::verdict::{GuardrailResult, GuardrailSource, GuardrailType, VerdictValue};

// ---------------------------------------------------------------------------
// EngineState — engine + per-request config, shared via Arc<EngineState>
// ---------------------------------------------------------------------------

/// Wraps the GuardrailEngine with config values that request handlers need.
/// This is the type stored in AppState.
pub struct EngineState {
    pub engine: GuardrailEngine,
    pub max_text_chars: usize,
    pub llm_base_url: String,
    pub llm_model: String,
    pub llm_api_key: String,
    pub python_executable: String,
}

impl std::ops::Deref for EngineState {
    type Target = GuardrailEngine;
    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

// ---------------------------------------------------------------------------
// GuardrailEngine
// ---------------------------------------------------------------------------

pub struct GuardrailEngine {
    pub base_models: Arc<BaseModelRegistry>,
    pub custom_models: Arc<CustomModelCache>,
    pub regex_base: Arc<BaseRegexScanner>,
    pub regex_custom_cache: Arc<CustomRegexCache>,
    pub models_dir: String,
}

impl GuardrailEngine {
    pub fn new(
        base_models: BaseModelRegistry,
        parapet_config: &str,
        models_dir: &str,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            base_models: Arc::new(base_models),
            custom_models: Arc::new(CustomModelCache::new()),
            regex_base: Arc::new(BaseRegexScanner::new(parapet_config)?),
            regex_custom_cache: Arc::new(CustomRegexCache::new()),
            models_dir: models_dir.to_string(),
        })
    }

    /// Run the full detect pipeline on normalized text for the selected guardrails.
    pub async fn run(
        &self,
        text: &str,
        selector: &GuardrailSelector,
        db: &DbPool,
    ) -> anyhow::Result<(Vec<GuardrailResult>, f32)> {
        let mut results: Vec<GuardrailResult> = Vec::new();
        let mut max_score: f32 = 0.0;

        // ── Base SVMs ────────────────────────────────────────────────────────
        let base_names = self.resolve_svm_base(&selector.svm_base);
        for name in &base_names {
            if let Some(model) = self.base_models.get(name) {
                let score = model.score(text);
                let verdict = if score >= 0.5 { VerdictValue::Block } else { VerdictValue::Allow };
                if score > max_score { max_score = score; }
                results.push(GuardrailResult {
                    guardrail_id: name.clone(),
                    guardrail_type: GuardrailType::Svm,
                    source: GuardrailSource::Base,
                    name: format!("{} SVM", name.replace('_', " ")),
                    category: Some(name.clone()),
                    description: Some(format!("Base SVM classifier for {} detection.", name)),
                    verdict,
                    score,
                    matched_patterns: None,
                });
            }
        }

        // ── Custom SVMs ──────────────────────────────────────────────────────
        let custom_model_ids = self.resolve_svm_custom(&selector.svm_custom, db).await?;
        for (id, model_meta) in &custom_model_ids {
            let path = match &model_meta.model_path {
                Some(p) => p.clone(),
                None => continue,
            };
            let model = self.custom_models.get(&id, &path).await?;
            let score = model.score(text);
            let verdict = if score >= 0.5 { VerdictValue::Block } else { VerdictValue::Allow };
            if score > max_score { max_score = score; }
            results.push(GuardrailResult {
                guardrail_id: id.clone(),
                guardrail_type: GuardrailType::Svm,
                source: GuardrailSource::Custom,
                name: model_meta.name.clone(),
                category: Some(model_meta.category.clone()),
                description: model_meta.description.clone(),
                verdict,
                score,
                matched_patterns: None,
            });
        }

        // ── Base Regex ────────────────────────────────────────────────────────
        let regex_cats = self.resolve_regex_base(&selector.regex_base);
        for cat in &regex_cats {
            let (score, matched) = self.regex_base.scan(text, &[cat.clone()]);
            let verdict = if score > 0.0 { VerdictValue::Block } else { VerdictValue::Allow };
            if score > max_score { max_score = score; }
            results.push(GuardrailResult {
                guardrail_id: cat.clone(),
                guardrail_type: GuardrailType::Regex,
                source: GuardrailSource::Base,
                name: BaseRegexScanner::category_name(cat),
                category: Some(cat.clone()),
                description: Some(BaseRegexScanner::category_description(cat)),
                verdict,
                score,
                matched_patterns: Some(matched),
            });
        }

        // ── Custom Regex ──────────────────────────────────────────────────────
        let custom_regex_groups =
            self.resolve_regex_custom(&selector.regex_custom, db).await?;
        for group in &custom_regex_groups {
            let (score, matched) = scan_custom_patterns(text, group);
            let verdict = if score > 0.0 { VerdictValue::Block } else { VerdictValue::Allow };
            if score > max_score { max_score = score; }
            results.push(GuardrailResult {
                guardrail_id: group.group_id.clone(),
                guardrail_type: GuardrailType::Regex,
                source: GuardrailSource::Custom,
                name: group.group_name.clone(),
                category: group.category.clone(),
                description: group.description.clone(),
                verdict,
                score,
                matched_patterns: Some(matched),
            });
        }

        Ok((results, max_score))
    }

    // ── Selector resolution ────────────────────────────────────────────────

    fn resolve_svm_base(&self, sel: &SvmBaseSelector) -> Vec<String> {
        match sel {
            SvmBaseSelector::All => vec!["allrounder".to_string()],
            SvmBaseSelector::Categories(cats) => cats.clone(),
            SvmBaseSelector::None => vec![],
        }
    }

    async fn resolve_svm_custom(
        &self,
        sel: &SvmCustomSelector,
        db: &DbPool,
    ) -> anyhow::Result<Vec<(String, CustomModelMeta)>> {
        match sel {
            SvmCustomSelector::None => Ok(vec![]),
            SvmCustomSelector::All => fetch_all_custom_models(db).await,
            SvmCustomSelector::Ids(ids) => fetch_custom_models_by_ids(ids, db).await,
        }
    }

    fn resolve_regex_base(&self, sel: &RegexBaseSelector) -> Vec<String> {
        use crate::engine::svm_base::BASE_MODEL_NAMES;
        match sel {
            RegexBaseSelector::None => vec![],
            RegexBaseSelector::All => BASE_MODEL_NAMES
                .iter()
                .filter(|&&n| n != "allrounder")
                .map(|s| s.to_string())
                .collect(),
            RegexBaseSelector::Categories(cats) => cats.clone(),
        }
    }

    async fn resolve_regex_custom(
        &self,
        sel: &RegexCustomSelector,
        db: &DbPool,
    ) -> anyhow::Result<Vec<PatternGroupRecord>> {
        match sel {
            RegexCustomSelector::None => Ok(vec![]),
            RegexCustomSelector::All => {
                fetch_all_custom_pattern_groups(db, &self.regex_custom_cache).await
            }
            RegexCustomSelector::Ids(ids) => {
                fetch_custom_pattern_groups_by_ids(ids, db, &self.regex_custom_cache).await
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lightweight DB row types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CustomModelMeta {
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub model_path: Option<String>,
}

// ---------------------------------------------------------------------------
// DB query helpers — dynamic queries (no DATABASE_URL needed at compile time)
// ---------------------------------------------------------------------------

async fn fetch_all_custom_models(db: &DbPool) -> anyhow::Result<Vec<(String, CustomModelMeta)>> {
    let sql = "SELECT id, name, description, category, model_path
               FROM custom_models WHERE status = 'ready' AND model_path IS NOT NULL";
    match db {
        DbPool::Sqlite(pool) => {
            let rows: Vec<(String, String, Option<String>, String, Option<String>)> =
                sqlx::query_as(sql).fetch_all(pool).await?;
            Ok(rows.into_iter().map(|(id, name, description, category, model_path)| {
                (id, CustomModelMeta { name, description, category, model_path })
            }).collect())
        }
        DbPool::Postgres(pool) => {
            let rows: Vec<(String, String, Option<String>, String, Option<String>)> =
                sqlx::query_as(sql).fetch_all(pool).await?;
            Ok(rows.into_iter().map(|(id, name, description, category, model_path)| {
                (id, CustomModelMeta { name, description, category, model_path })
            }).collect())
        }
    }
}

async fn fetch_custom_models_by_ids(
    ids: &[String],
    db: &DbPool,
) -> anyhow::Result<Vec<(String, CustomModelMeta)>> {
    let mut results = Vec::new();
    for id in ids {
        let sql = match db {
            DbPool::Sqlite(_) => "SELECT id, name, description, category, model_path FROM custom_models WHERE id = ?",
            DbPool::Postgres(_) => "SELECT id, name, description, category, model_path FROM custom_models WHERE id = $1",
        };
        let row: Option<(String, String, Option<String>, String, Option<String>)> = match db {
            DbPool::Sqlite(pool) => sqlx::query_as(sql).bind(id).fetch_optional(pool).await?,
            DbPool::Postgres(pool) => sqlx::query_as(sql).bind(id).fetch_optional(pool).await?,
        };
        if let Some((id, name, description, category, model_path)) = row {
            results.push((id, CustomModelMeta { name, description, category, model_path }));
        }
    }
    Ok(results)
}

async fn fetch_all_custom_pattern_groups(
    db: &DbPool,
    cache: &CustomRegexCache,
) -> anyhow::Result<Vec<PatternGroupRecord>> {
    let group_ids: Vec<String> = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query_as::<_, (String,)>("SELECT id FROM pattern_groups")
                .fetch_all(pool).await?.into_iter().map(|(id,)| id).collect()
        }
        DbPool::Postgres(pool) => {
            sqlx::query_as::<_, (String,)>("SELECT id FROM pattern_groups")
                .fetch_all(pool).await?.into_iter().map(|(id,)| id).collect()
        }
    };
    let mut out = Vec::new();
    for id in &group_ids {
        out.push(load_pattern_group(id, db, cache).await?);
    }
    Ok(out)
}

async fn fetch_custom_pattern_groups_by_ids(
    ids: &[String],
    db: &DbPool,
    cache: &CustomRegexCache,
) -> anyhow::Result<Vec<PatternGroupRecord>> {
    let mut out = Vec::new();
    for id in ids {
        out.push(load_pattern_group(id, db, cache).await?);
    }
    Ok(out)
}

async fn load_pattern_group(
    group_id: &str,
    db: &DbPool,
    cache: &CustomRegexCache,
) -> anyhow::Result<PatternGroupRecord> {
    if let Some(cached) = cache.get(group_id).await {
        return Ok(cached);
    }

    let (group_name, description, category): (String, Option<String>, Option<String>) = match db {
        DbPool::Sqlite(pool) => sqlx::query_as(
            "SELECT name, description, category FROM pattern_groups WHERE id = ?"
        ).bind(group_id).fetch_one(pool).await?,
        DbPool::Postgres(pool) => sqlx::query_as(
            "SELECT name, description, category FROM pattern_groups WHERE id = $1"
        ).bind(group_id).fetch_one(pool).await?,
    };

    let raw_patterns: Vec<String> = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query_as::<_, (String,)>(
                "SELECT pattern FROM pattern_entries WHERE group_id = ?"
            ).bind(group_id).fetch_all(pool).await?
            .into_iter().map(|(p,)| p).collect()
        }
        DbPool::Postgres(pool) => {
            sqlx::query_as::<_, (String,)>(
                "SELECT pattern FROM pattern_entries WHERE group_id = $1"
            ).bind(group_id).fetch_all(pool).await?
            .into_iter().map(|(p,)| p).collect()
        }
    };

    let mut patterns = Vec::new();
    for raw in raw_patterns {
        match compile_pattern(&raw) {
            Ok(compiled) => patterns.push((raw, compiled)),
            Err(e) => tracing::warn!(
                group_id,
                pattern = raw,
                error = e,
                "Skipping invalid regex in custom pattern group"
            ),
        }
    }

    let record = PatternGroupRecord {
        group_id: group_id.to_string(),
        group_name,
        description,
        category,
        patterns,
    };
    cache.insert(record.clone()).await;
    Ok(record)
}
