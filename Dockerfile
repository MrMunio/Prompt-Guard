# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM rust:1.82-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy Rust library dependency and standalone project files
COPY parapet/ ./parapet/
COPY parapet-guardrail/ ./parapet-guardrail/

WORKDIR /build/parapet-guardrail
RUN cargo build --release

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install Python ML dependencies for training scripts
RUN pip install --no-cache-dir scikit-learn numpy httpx pyyaml

# Copy Rust binary
COPY --from=builder /build/parapet-guardrail/target/release/parapet-guardrail /app/parapet-guardrail

# Copy training scripts, dataset schemas, and default parapet.yaml
COPY parapet-guardrail/scripts/ /app/scripts/
COPY parapet-guardrail/schema/ /app/schema/
COPY parapet-guardrail/parapet.yaml /app/parapet.yaml

# Create models directory structure
RUN mkdir -p /app/models/base /app/models/custom /app/data

EXPOSE 9900

CMD ["/app/parapet-guardrail"]
