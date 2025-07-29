#!/bin/bash

# Synth-AI Environment Service Startup Script
# Starts the environment service on port 8901 for Crafter and other environments

set -e

echo "🚀 Starting Synth-AI Environment Service..."
echo "   Port: 8901"
echo "   Host: 0.0.0.0"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "synth_ai" ]; then
    echo "❌ Error: Must run from synth-ai project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: pyproject.toml, synth_ai/"
    exit 1
fi

# Set environment variables for better logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export SYNTH_LOGGING="true"

# Start the service
echo "📡 Starting uvicorn server..."
echo "   Access at: http://localhost:8901"
echo "   Health check: http://localhost:8901/health"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

uv run python -m uvicorn \
    synth_ai.environments.service.app:app \
    --host 0.0.0.0 \
    --port 8901 \
    --log-level info \
    --reload \
    --reload-dir synth_ai
