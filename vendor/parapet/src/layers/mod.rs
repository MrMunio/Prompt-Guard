// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Layer implementations for the Guardrail detection pipeline:
//   L1  - char n-gram SVM classifier (lightweight, compiled-in weights)
//   L3  - regex/pattern-based block and evidence scanning
//   L4  - multi-turn risk scoring (kept for lib completeness; not called by the API)
//
// Archived layers (l2a, l2_semantic, l5a) have been moved to _archive/.

pub mod l1;
pub mod l1_harness;
pub mod l3_inbound;
pub mod l4;

// Stub l2a_model to satisfy config load validation without ONNX dependencies.
pub mod l2a_model {
    pub const KNOWN_MODELS: &[&str] = &["prompt-guard-86m-v2"];
}


