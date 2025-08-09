#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "🧪 Running integration tests (integration/slow markers)"
uv run pytest -c pytest.integration.ini -m "integration or slow"
