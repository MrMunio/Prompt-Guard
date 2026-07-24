// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Built-in regex scanner wrapper.
//!
//! Wraps parapet's `DefaultInboundScanner` (L3) but filters by category.
//!
//! The existing parapet L3 patterns don't yet have a `category` field in their
//! compiled representation. Until Phase 5 (category tagging audit) is complete,
//! this module scans ALL patterns when any base category is requested, and
//! returns all matches. Phase 5 will add a proper category→pattern-index map.

use parapet::config::{load_config, Config, FileSource, StringSource};
use parapet::layers::l3_inbound::{DefaultInboundScanner, InboundScanner};
use parapet::message::{Message, Role, TrustLevel};

// ---------------------------------------------------------------------------
// BaseRegexScanner
// ---------------------------------------------------------------------------

pub struct BaseRegexScanner {
    config: Config,
    scanner: DefaultInboundScanner,
}

impl BaseRegexScanner {
    /// Load the parapet config from `parapet_config_path`. Falls back to a
    /// minimal default config if the path does not exist, so the service
    /// starts up even without a `parapet.yaml` beside the binary.
    pub fn new(parapet_config_path: &str) -> anyhow::Result<Self> {
        let config = if std::path::Path::new(parapet_config_path).exists() {
            let source = FileSource {
                path: std::path::PathBuf::from(parapet_config_path),
            };
            load_config(&source).map_err(|e| anyhow::anyhow!("Failed to load parapet config: {e}"))?
        } else {
            tracing::warn!(
                path = parapet_config_path,
                "parapet.yaml not found — using built-in default config for L3 patterns"
            );
            let source = StringSource { content: "parapet: v1\n".to_string() };
            load_config(&source).map_err(|e| anyhow::anyhow!("Failed to load default config: {e}"))?
        };

        Ok(Self {
            config,
            scanner: DefaultInboundScanner::new(),
        })
    }

    /// Scan text against base patterns filtered to the requested categories.
    ///
    /// `categories` is a list of category names to filter. If empty, scans all.
    /// Returns `(score, matched_pattern_strings)`.
    ///
    /// NOTE (Phase 5 TODO): categories are not yet mapped to pattern indices;
    /// currently all patterns are scanned regardless of category filter.
    pub fn scan(&self, text: &str, _categories: &[String]) -> (f32, Vec<String>) {
        // Wrap the text as a single untrusted user message.
        let mut msg = Message::new(Role::User, text);
        msg.trust = TrustLevel::Untrusted;

        let result = self.scanner.scan(&[msg], &self.config);

        let mut matched = Vec::new();
        for pm in &result.matched_patterns {
            if let Some(p) = self.config.policy.block_patterns.get(pm.pattern_index) {
                matched.push(p.pattern.clone());
            }
        }

        let score = if matched.is_empty() { 0.0 } else { 1.0 };
        (score, matched)
    }

    /// Returns the display name for a given base regex category.
    pub fn category_name(category: &str) -> String {
        format!("{} Patterns", category.replace('_', " "))
    }

    /// Returns a short description for a given base regex category.
    pub fn category_description(category: &str) -> String {
        match category {
            "instruction_override" => "Built-in patterns detecting instruction override attempts.",
            "roleplay_jailbreak"   => "Built-in patterns detecting roleplay jailbreak attempts.",
            "meta_probe"           => "Built-in patterns detecting system prompt probing.",
            "exfiltration"         => "Built-in patterns detecting data exfiltration attempts.",
            "adversarial_suffix"   => "Built-in patterns detecting adversarial suffix attacks.",
            "indirect_injection"   => "Built-in patterns detecting indirect injection via documents.",
            "obfuscation"          => "Built-in patterns detecting obfuscated attacks.",
            "constraint_bypass"    => "Built-in patterns detecting policy constraint bypass.",
            _                      => "Built-in regex pattern set.",
        }.to_string()
    }
}
