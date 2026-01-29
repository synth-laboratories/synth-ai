# Dockerfile for Harbor EngineBench Runner
# This image runs inside Daytona sandboxes for coding agent evaluations.
#
# The runner follows the Harbor contract:
#   Input:  /tmp/rollout.json (trace_correlation_id, seed, prompt_template, inference_url, limits)
#   Output: /tmp/result.json  (trace_correlation_id, metrics.reward_mean, success)
#
FROM python:3.11-slim

# Install system deps + Rust + Node.js (for Codex/OpenCode agents)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ca-certificates \
    gnupg \
    unzip \
    jq \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

# Install Codex CLI globally
RUN npm install -g @openai/codex

# Install OpenCode CLI
RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="/root/.opencode/bin:$PATH"

# Install Claude Code CLI (Anthropic)
RUN npm install -g @anthropic-ai/claude-code

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python deps
RUN uv pip install --system fastapi uvicorn httpx pydantic openai

# Clone engine-bench repo (needed for cargo tests + instance data)
RUN git clone https://github.com/JoshuaPurtell/engine-bench.git /engine-bench

# Pre-download Rust dependencies (speeds up first cargo check/test)
WORKDIR /engine-bench
RUN cargo fetch 2>/dev/null || true
RUN cargo build --release 2>/dev/null || true

# Verify instance data is present (non-fatal if data dir structure differs)
RUN ls /engine-bench/data/instances/single/*.json 2>/dev/null | wc -l && \
    echo "Instance data check complete"

# Pre-create config directories
RUN mkdir -p /root/.codex /root/.opencode /app /tmp

WORKDIR /app

# Copy runner script - version 2 (v1 URL fix)
COPY run_rollout.py /app/run_rollout.py
RUN chmod +x /app/run_rollout.py

# Create the run_rollout command wrapper
RUN echo '#!/bin/bash\npython3 /app/run_rollout.py "$@"' > /usr/local/bin/run_rollout && \
    chmod +x /usr/local/bin/run_rollout

# Verify setup
RUN python3 -c "import json, subprocess; print('Python deps OK')"
RUN rustc --version && cargo --version
RUN codex --version
RUN opencode --version
RUN claude --version
RUN node --version

# Default command shows help
CMD ["run_rollout", "--help"]
