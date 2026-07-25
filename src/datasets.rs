// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Dataset catalog seeding — runs once at startup (async, non-blocking).
//!
//! Upserts the known open-source dataset catalog into training_datasets,
//! then scans each file that is expected to be "ready" and counts its
//! records (attack_count, benign_count, record_count).
//!
//! The scan uses a simple line-count heuristic on the YAML (counts "- id:"
//! lines) to avoid loading full YAML at startup cost. An exact count is
//! not critical here — it is display-only metadata for the UI.

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;

use crate::db::DbPool;

// ---------------------------------------------------------------------------
// Static catalog (all known datasets)
// ---------------------------------------------------------------------------

struct CatalogEntry {
    id:           &'static str,
    file_name:    &'static str,
    display_name: &'static str,
    description:  &'static str,
    category:     &'static str,
    label_type:   &'static str,
    fetch_status: &'static str,  // "ready" | "fetchable" | "private"
    hf_uri:       &'static str,
    source_url:   &'static str,
    license:      &'static str,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "l1_attacks",
        file_name: "l1_attacks.yaml",
        display_name: "Parapet Curated Attacks",
        description: "Curated prompt injection attacks from the Parapet project.",
        category: "instruction_override",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "",
        source_url: "",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "l1_benign",
        file_name: "l1_benign.yaml",
        display_name: "Parapet Curated Benign",
        description: "Curated benign prompts from the Parapet project.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "",
        source_url: "",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_gandalf_attacks",
        file_name: "opensource_gandalf_attacks.yaml",
        display_name: "Gandalf Ignore Instructions",
        description: "1,000+ instruction override attacks from Lakera's Gandalf game.",
        category: "instruction_override",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "Lakera/gandalf_ignore_instructions",
        source_url: "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        license: "CC-BY-4.0",
    },
    CatalogEntry {
        id: "opensource_giskard_attacks",
        file_name: "opensource_giskard_attacks.yaml",
        display_name: "Giskard Prompt Injections",
        description: "35 curated DAN/hijacking attacks from Giskard-AI (garak + PromptInject).",
        category: "meta_probe",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "",
        source_url: "https://github.com/Giskard-AI/prompt-injections",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_hackaprompt_attacks",
        file_name: "opensource_hackaprompt_attacks.yaml",
        display_name: "HackAPrompt Dataset",
        description: "2,000+ constraint bypass attacks from the HackAPrompt competition.",
        category: "constraint_bypass",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "hackaprompt/hackaprompt-dataset",
        source_url: "https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset",
        license: "CC-BY-4.0",
    },
    CatalogEntry {
        id: "opensource_jailbreak_cls_attacks",
        file_name: "opensource_jailbreak_cls_attacks.yaml",
        display_name: "Jailbreak Classification Attacks",
        description: "10,000+ roleplay jailbreaks from jackhhao/jailbreak-classification.",
        category: "roleplay_jailbreak",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "jackhhao/jailbreak-classification",
        source_url: "https://huggingface.co/datasets/jackhhao/jailbreak-classification",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_chatgpt_jailbreak_attacks",
        file_name: "opensource_chatgpt_jailbreak_attacks.yaml",
        display_name: "ChatGPT Jailbreak Prompts",
        description: "1,500+ roleplay/instruction override attacks from rubend18's collection.",
        category: "roleplay_jailbreak",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "rubend18/ChatGPT-Jailbreak-Prompts",
        source_url: "https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts",
        license: "CC0-1.0",
    },
    CatalogEntry {
        id: "opensource_deepset_attacks",
        file_name: "opensource_deepset_attacks.yaml",
        display_name: "Deepset Prompt Injections",
        description: "800 instruction override attacks from deepset/prompt-injections.",
        category: "instruction_override",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "deepset/prompt-injections",
        source_url: "https://huggingface.co/datasets/deepset/prompt-injections",
        license: "CC-BY-4.0",
    },
    CatalogEntry {
        id: "opensource_mosscap_attacks",
        file_name: "opensource_mosscap_attacks.yaml",
        display_name: "Mosscap Jailbreaks",
        description: "5,000 roleplay/constraint bypass jailbreaks.",
        category: "roleplay_jailbreak",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "",
        source_url: "",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_wildguardmix_attacks",
        file_name: "opensource_wildguardmix_attacks.yaml",
        display_name: "WildGuardMix Attacks",
        description: "10,000+ constraint bypass and roleplay attacks from allenai/wildguardmix.",
        category: "constraint_bypass",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "allenai/wildguardmix",
        source_url: "https://huggingface.co/datasets/allenai/wildguardmix",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_jbb_attacks",
        file_name: "opensource_jbb_attacks.yaml",
        display_name: "JailbreakBench Attacks",
        description: "200 adversarial and constraint bypass attacks from JailbreakBench.",
        category: "constraint_bypass",
        label_type: "attack",
        fetch_status: "ready",
        hf_uri: "",
        source_url: "https://github.com/JailbreakBench/jailbreakbench",
        license: "MIT",
    },
    CatalogEntry {
        id: "opensource_notinject_benign",
        file_name: "opensource_notinject_benign.yaml",
        display_name: "NotInject Benign",
        description: "1,000+ benign prompts from leolee99/NotInject.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "leolee99/NotInject",
        source_url: "https://huggingface.co/datasets/leolee99/NotInject",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_no_robots_benign",
        file_name: "opensource_no_robots_benign.yaml",
        display_name: "No Robots Benign",
        description: "10,000+ benign instruction-following prompts.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "HuggingFaceH4/no_robots",
        source_url: "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
        license: "CC-BY-NC-4.0",
    },
    CatalogEntry {
        id: "opensource_wildguardmix_benign",
        file_name: "opensource_wildguardmix_benign.yaml",
        display_name: "WildGuardMix Benign",
        description: "10,000+ benign prompts from allenai/wildguardmix.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "allenai/wildguardmix",
        source_url: "https://huggingface.co/datasets/allenai/wildguardmix",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_chatgpt_prompts_benign",
        file_name: "opensource_chatgpt_prompts_benign.yaml",
        display_name: "ChatGPT Prompts Benign",
        description: "1,500 benign ChatGPT prompts.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "MohamedRashad/ChatGPT-prompts",
        source_url: "https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts",
        license: "CC0-1.0",
    },
    CatalogEntry {
        id: "opensource_jailbreak_cls_benign",
        file_name: "opensource_jailbreak_cls_benign.yaml",
        display_name: "Jailbreak Classification Benign",
        description: "5,000+ benign prompts from jackhhao/jailbreak-classification.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "jackhhao/jailbreak-classification",
        source_url: "https://huggingface.co/datasets/jackhhao/jailbreak-classification",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_deepset_benign",
        file_name: "opensource_deepset_benign.yaml",
        display_name: "Deepset Benign",
        description: "800 benign prompts from deepset/prompt-injections.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "deepset/prompt-injections",
        source_url: "https://huggingface.co/datasets/deepset/prompt-injections",
        license: "CC-BY-4.0",
    },
    CatalogEntry {
        id: "opensource_jbb_benign",
        file_name: "opensource_jbb_benign.yaml",
        display_name: "JailbreakBench Benign",
        description: "200 benign prompts from JailbreakBench.",
        category: "general",
        label_type: "benign",
        fetch_status: "ready",
        hf_uri: "",
        source_url: "https://github.com/JailbreakBench/jailbreakbench",
        license: "MIT",
    },
    // ── Fetchable datasets (not yet downloaded) ─────────────────────────────
    CatalogEntry {
        id: "opensource_bipia_attacks",
        file_name: "opensource_bipia_attacks.yaml",
        display_name: "BIPIA Indirect Injection",
        description: "1,600 indirect injection attacks embedded in emails/tables/code contexts.",
        category: "indirect_injection",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "",
        source_url: "https://github.com/microsoft/BIPIA",
        license: "MIT",
    },
    CatalogEntry {
        id: "opensource_imoxto_attacks",
        file_name: "opensource_imoxto_attacks.yaml",
        display_name: "Imoxto Prompt Injection v2",
        description: "1,000+ roleplay and exfiltration attacks.",
        category: "exfiltration",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "imoxto/prompt_injection_cleaned_dataset-v2",
        source_url: "https://huggingface.co/datasets/imoxto/prompt_injection_cleaned_dataset-v2",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_jailbreakv_attacks",
        file_name: "opensource_jailbreakv_attacks.yaml",
        display_name: "JailbreakV-28k",
        description: "2,000 sampled roleplay jailbreaks from JailbreakV-28K.",
        category: "roleplay_jailbreak",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "JailbreakV-28K/JailBreakV-28k",
        source_url: "https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k",
        license: "Apache-2.0",
    },
    CatalogEntry {
        id: "opensource_llmail_attacks",
        file_name: "opensource_llmail_attacks.yaml",
        display_name: "LLMail Inject Challenge",
        description: "1,289 indirect injection attacks from Microsoft's LLMail challenge.",
        category: "indirect_injection",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "microsoft/llmail-inject-challenge",
        source_url: "https://huggingface.co/datasets/microsoft/llmail-inject-challenge",
        license: "MIT",
    },
    CatalogEntry {
        id: "opensource_tensortrust_hijacking_attacks",
        file_name: "opensource_tensortrust_hijacking_attacks.yaml",
        display_name: "TensorTrust Hijacking",
        description: "2,000 instruction override attacks from the TensorTrust game.",
        category: "instruction_override",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "",
        source_url: "https://github.com/HumanCompatibleAI/tensor-trust-data",
        license: "MIT",
    },
    CatalogEntry {
        id: "opensource_tensortrust_extraction_attacks",
        file_name: "opensource_tensortrust_extraction_attacks.yaml",
        display_name: "TensorTrust Extraction",
        description: "2,000 meta-probe attacks (system prompt extraction) from TensorTrust.",
        category: "meta_probe",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "",
        source_url: "https://github.com/HumanCompatibleAI/tensor-trust-data",
        license: "MIT",
    },
    CatalogEntry {
        id: "opensource_jbb_paraphrase_attacks",
        file_name: "opensource_jbb_paraphrase_attacks.yaml",
        display_name: "JailbreakBench Paraphrase",
        description: "500 paraphrased constraint bypass attacks.",
        category: "constraint_bypass",
        label_type: "attack",
        fetch_status: "fetchable",
        hf_uri: "DhruvTre/jailbreakbench-paraphrase-2025-08",
        source_url: "https://huggingface.co/datasets/DhruvTre/jailbreakbench-paraphrase-2025-08",
        license: "MIT",
    },
    CatalogEntry {
        id: "opensource_alpaca_benign",
        file_name: "opensource_alpaca_benign.yaml",
        display_name: "Alpaca Instructions",
        description: "Benign instruction-following prompts from tatsu-lab/alpaca.",
        category: "general",
        label_type: "benign",
        fetch_status: "fetchable",
        hf_uri: "tatsu-lab/alpaca",
        source_url: "https://huggingface.co/datasets/tatsu-lab/alpaca",
        license: "CC-BY-NC-4.0",
    },
    CatalogEntry {
        id: "opensource_hc3_benign",
        file_name: "opensource_hc3_benign.yaml",
        display_name: "HC3 Human-ChatGPT",
        description: "Benign human-written prompts from Hello-SimpleAI/HC3.",
        category: "general",
        label_type: "benign",
        fetch_status: "fetchable",
        hf_uri: "Hello-SimpleAI/HC3",
        source_url: "https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        license: "CC-BY-SA-4.0",
    },
    // ── Private datasets (curated, not publicly available) ───────────────────
    CatalogEntry {
        id: "obfuscation_private",
        file_name: "thewall_obfuscation.yaml",
        display_name: "Parapet Obfuscation Curated (Private)",
        description: "Private curated obfuscation dataset. Not available for client blending.",
        category: "obfuscation",
        label_type: "attack",
        fetch_status: "private",
        hf_uri: "",
        source_url: "",
        license: "",
    },
];

// ---------------------------------------------------------------------------
// Startup seeder
// ---------------------------------------------------------------------------

/// Seed the training_datasets catalog and scan file sizes.
/// Runs asynchronously — does not block server startup.
pub async fn seed_dataset_catalog(db: Arc<DbPool>, schema_eval_dir: String) {
    tokio::spawn(async move {
        if let Err(e) = do_seed(&db, &schema_eval_dir).await {
            tracing::warn!(error = %e, "Dataset catalog seeding failed (non-fatal)");
        }
    });
}

async fn do_seed(db: &DbPool, schema_eval_dir: &str) -> anyhow::Result<()> {
    let now = Utc::now().to_rfc3339();
    let eval_dir = Path::new(schema_eval_dir);

    let mut seeded = 0usize;
    let mut updated = 0usize;

    for entry in CATALOG {
        // Determine file path and effective fetch_status.
        let file_path = eval_dir.join(entry.file_name);
        let (effective_status, path_str, record_count, attack_count, benign_count) =
            if file_path.exists() {
                let (rc, ac, bc) = count_yaml_records(&file_path);
                (
                    "ready",
                    Some(file_path.to_string_lossy().into_owned()),
                    Some(rc as i64),
                    Some(ac as i64),
                    Some(bc as i64),
                )
            } else if entry.fetch_status == "private" {
                ("private", None, None, None, None)
            } else {
                (entry.fetch_status, None, None, None, None)
            };

        let rows_affected = upsert_entry(
            db, entry, effective_status,
            path_str.as_deref(), record_count, attack_count, benign_count, &now,
        ).await?;

        if rows_affected > 0 { seeded += 1; } else { updated += 1; }
    }

    tracing::info!(
        seeded, updated,
        total = CATALOG.len(),
        "Dataset catalog seeded"
    );
    Ok(())
}

/// Count records in a YAML file by counting lines that start with "- " (list items)
/// or contain "id:" as a proxy. Fast, no full YAML parse needed for display counts.
fn count_yaml_records(path: &Path) -> (usize, usize, usize) {
    use std::io::{BufRead, BufReader};
    let Ok(file) = std::fs::File::open(path) else { return (0, 0, 0); };
    let reader = BufReader::new(file);

    let mut total = 0usize;
    let mut attack = 0usize;
    let mut benign = 0usize;

    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("- id:") || trimmed.starts_with("id:") {
            total += 1;
        }
        if trimmed.contains("label: malicious") || trimmed.contains("\"label\": 1") {
            attack += 1;
        }
        if trimmed.contains("label: benign") || trimmed.contains("\"label\": 0") {
            benign += 1;
        }
    }

    // If no explicit label lines found, infer from file name.
    if attack == 0 && benign == 0 && total > 0 {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.contains("_attacks") { attack = total; }
        else if name.contains("_benign") { benign = total; }
    }

    (total, attack, benign)
}

async fn upsert_entry(
    db: &DbPool,
    entry: &CatalogEntry,
    status: &str,
    file_path: Option<&str>,
    record_count: Option<i64>,
    attack_count: Option<i64>,
    benign_count: Option<i64>,
    now: &str,
) -> anyhow::Result<u64> {
    let rows = match db {
        DbPool::Sqlite(pool) => {
            sqlx::query(
                "INSERT INTO training_datasets \
                 (id, file_name, display_name, description, category, label_type, \
                  record_count, attack_count, benign_count, file_path, fetch_status, \
                  hf_uri, source_url, license, last_indexed_at) \
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) \
                 ON CONFLICT(id) DO UPDATE SET \
                  file_path=excluded.file_path, \
                  fetch_status=excluded.fetch_status, \
                  record_count=excluded.record_count, \
                  attack_count=excluded.attack_count, \
                  benign_count=excluded.benign_count, \
                  last_indexed_at=excluded.last_indexed_at"
            )
            .bind(entry.id)
            .bind(entry.file_name)
            .bind(entry.display_name)
            .bind(if entry.description.is_empty() { None } else { Some(entry.description) })
            .bind(entry.category)
            .bind(entry.label_type)
            .bind(record_count)
            .bind(attack_count)
            .bind(benign_count)
            .bind(file_path)
            .bind(status)
            .bind(if entry.hf_uri.is_empty() { None } else { Some(entry.hf_uri) })
            .bind(if entry.source_url.is_empty() { None } else { Some(entry.source_url) })
            .bind(if entry.license.is_empty() { None } else { Some(entry.license) })
            .bind(now)
            .execute(pool)
            .await?
            .rows_affected()
        }
        DbPool::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO training_datasets \
                 (id, file_name, display_name, description, category, label_type, \
                  record_count, attack_count, benign_count, file_path, fetch_status, \
                  hf_uri, source_url, license, last_indexed_at) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15) \
                 ON CONFLICT(id) DO UPDATE SET \
                  file_path=EXCLUDED.file_path, \
                  fetch_status=EXCLUDED.fetch_status, \
                  record_count=EXCLUDED.record_count, \
                  attack_count=EXCLUDED.attack_count, \
                  benign_count=EXCLUDED.benign_count, \
                  last_indexed_at=EXCLUDED.last_indexed_at"
            )
            .bind(entry.id)
            .bind(entry.file_name)
            .bind(entry.display_name)
            .bind(if entry.description.is_empty() { None } else { Some(entry.description) })
            .bind(entry.category)
            .bind(entry.label_type)
            .bind(record_count)
            .bind(attack_count)
            .bind(benign_count)
            .bind(file_path)
            .bind(status)
            .bind(if entry.hf_uri.is_empty() { None } else { Some(entry.hf_uri) })
            .bind(if entry.source_url.is_empty() { None } else { Some(entry.source_url) })
            .bind(if entry.license.is_empty() { None } else { Some(entry.license) })
            .bind(now)
            .execute(pool)
            .await?
            .rows_affected()
        }
    };
    Ok(rows)
}
