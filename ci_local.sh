#!/usr/bin/env bash
set -euo pipefail

# Local CI runner to mirror GitHub Actions steps
# - Lint (ruff)
# - Type Check (ty)
# - Tests: fast unit subset (≤5s), plugin autoload disabled

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[ci_local] Syncing dev dependencies with uv..."
uv sync --dev

echo "[ci_local] Running ruff..."
uvx --from ruff ruff check synth_ai examples

echo "[ci_local] Running ty check..."
uvx --from ty ty check

echo "[ci_local] Running fast unit tests (≤5s gate)..."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
SECONDS=0
uv run pytest tests/unit -m unit -q --maxfail=1
uv run pytest tests/environments/unit/test_external_registry.py -q --maxfail=1
DUR=$SECONDS
echo "[ci_local] Total fast unit test time: ${DUR}s"
if [ "$DUR" -gt 5 ]; then
  echo "[ci_local] Fast unit tests exceeded 5 seconds (${DUR}s)" >&2
  exit 1
fi

echo "[ci_local] Success."

