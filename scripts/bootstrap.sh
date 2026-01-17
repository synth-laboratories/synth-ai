#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_REPO="$HOME/synth-ai/main"

# Clone .env files from main repo
if [ -d "$MAIN_REPO" ]; then
  echo "Cloning .env files from $MAIN_REPO..."

  # Root .env files
  for env_file in "$MAIN_REPO"/.env*; do
    if [ -f "$env_file" ]; then
      cp "$env_file" "$ROOT/"
      echo "  Copied $(basename "$env_file") to root"
    fi
  done
else
  echo "Warning: Main repo not found at $MAIN_REPO, skipping .env clone"
fi

# Python (uv) at root
if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/pyproject.toml" ]; then
  echo "Setting up Python environment..."
  cd "$ROOT"
  uv sync
fi

# Node (bun) at root
if command -v bun >/dev/null 2>&1 && [ -f "$ROOT/package.json" ]; then
  echo "Installing Node dependencies at root..."
  cd "$ROOT"
  bun install
fi

# Node (bun) in synth_ai/tui/app/
if command -v bun >/dev/null 2>&1 && [ -f "$ROOT/synth_ai/tui/app/package.json" ]; then
  echo "Installing Node dependencies in synth_ai/tui/app/..."
  cd "$ROOT/synth_ai/tui/app"
  bun install
fi

echo "Bootstrap complete"
