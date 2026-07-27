// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0
//
// Public API surface for the parapet core detection library.
//
// Modules kept: config, constraint, layers (L1/L3), message, normalize,
//               signal, trust.
//
// Archived modules (engine, proxy, provider, stream, routing, session,
// sensor, defang, model_fetch, model_path) have been moved to _archive/.

pub mod config;
pub mod constraint;
pub mod layers;
pub mod message;
pub mod normalize;
pub mod signal;
pub mod trust;
