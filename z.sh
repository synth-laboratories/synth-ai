#!/bin/bash

set -e

echo ''
echo "Clearing uv cache"
rm -rf ~/.cache/uv
echo "Cleared uv cache"

echo ''
echo "Installing synth-ai"
uv pip install -e .
echo "Installed synth-ai"

echo ''
echo "Activating venv"
source .venv/bin/activate
echo "Activated venv"

echo ''
echo ":)"
echo ''
