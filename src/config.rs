// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Application configuration — loaded from environment variables (via .env file).
//!
//! All fields are read once at startup. Missing required fields cause a panic
//! with a clear error message rather than silent misconfiguration.

use std::env;

// ---------------------------------------------------------------------------
// Config struct
// ---------------------------------------------------------------------------

/// Top-level application config loaded from environment variables.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP port to listen on.
    pub port: u16,

    /// API key required in X-API-Key header for all protected routes.
    pub api_key: String,

    /// SQLx database URL (sqlite:... or postgres://...).
    pub database_url: String,

    /// LLM API base URL (OpenAI-compatible).
    pub llm_base_url: String,

    /// LLM model name (default: gpt-4o-mini).
    pub llm_model: String,

    /// LLM API key.
    pub llm_api_key: String,

    /// Maximum allowed input text length in characters.
    pub max_text_chars: usize,

    /// Directory where base and custom model weight files are stored.
    pub models_dir: String,

    /// Path to parapet.yaml (for L3 built-in pattern config).
    pub parapet_config: String,

    /// Path to Python executable (e.g., 'python', 'C:\Users\USER\.conda\envs\ml-guardrails\python.exe').
    pub python_executable: String,
}

impl AppConfig {
    /// Load config from environment. Panics on missing required fields.
    pub fn from_env() -> Self {
        // Load .env file from current working directory or subfolders
        let _ = dotenvy::dotenv().or_else(|_| dotenvy::from_filename("parapet-guardrail/.env"));

        Self {
            port: env_parse("PORT", 9900),
            api_key: env_require("API_KEY"),
            database_url: env_require("DATABASE_URL"),
            llm_base_url: env_default("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model: env_default("LLM_MODEL", "gpt-4o-mini"),
            llm_api_key: env_default("LLM_API_KEY", ""),
            max_text_chars: env_parse("MAX_TEXT_CHARS", 500_000),
            models_dir: env_default("MODELS_DIR", "./models"),
            parapet_config: env_default("PARAPET_CONFIG", "./parapet.yaml"),
            python_executable: env_default("PYTHON_EXECUTABLE", "python"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn env_require(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("Required environment variable '{key}' is not set"))
}

fn env_default(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr + std::fmt::Display>(key: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    match env::var(key) {
        Ok(val) => val
            .parse()
            .unwrap_or_else(|_| panic!("Environment variable '{key}' has invalid value '{val}'")),
        Err(_) => default,
    }
}
