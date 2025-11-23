#!/bin/bash
# Publish synth-ai to PyPI
# Automatically finds PYPI_API_KEY from monorepo/backend/.env.dev

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Look for PyPI token in common locations
ENV_FILE="$HOME/Documents/GitHub/monorepo/.env.dev"

if [ -z "$PYPI_API_KEY" ]; then
    if [ -f "$ENV_FILE" ]; then
        export PYPI_API_KEY=$(grep -E "^PYPI_API_KEY=" "$ENV_FILE" | cut -d= -f2-)
    fi
fi

if [ -z "$PYPI_API_KEY" ]; then
    echo "Error: PYPI_API_KEY not found"
    echo "Either:"
    echo "  1. Set PYPI_API_KEY environment variable"
    echo "  2. Add PYPI_API_KEY=pypi-xxx to $ENV_FILE"
    exit 1
fi

echo "Building synth-ai..."
cd "$REPO_ROOT"

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build
uv run python -m build

echo "Publishing to PyPI..."
uv run python -m twine upload dist/* -u __token__ -p "$PYPI_API_KEY"

echo "Done! Published version:"
grep "^version" pyproject.toml
