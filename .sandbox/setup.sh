#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace/synth-ai}

cd "$WORKSPACE_DIR"

git config --global --add safe.directory "$WORKSPACE_DIR" >/dev/null 2>&1 || true

if command -v uv >/dev/null 2>&1; then
  uv sync --frozen
  # synth-ai uses a Rust extension via maturin (PyO3).
  maturin develop --release --uv
else
  echo "uv not found; cannot install Python deps for synth-ai" >&2
  exit 1
fi
