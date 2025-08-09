#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "🧪 Running unit tests (fast/unit markers)"
uv run pytest -c pytest.unit.ini -m "fast or unit"
