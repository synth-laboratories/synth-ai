#!/bin/bash
# Deploy Banking77 task app locally for GEPA optimization

set -e

echo "🚀 Deploying Banking77 Task App..."
echo "=================================="

# Set up environment variables
export ENVIRONMENT_API_KEY="${ENVIRONMENT_API_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export GROQ_API_KEY="${GROQ_API_KEY}"

# Check for required env vars
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY not set"
    echo "Please set it: export GROQ_API_KEY=your_key"
    exit 1
fi

echo "✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
echo "✅ GROQ_API_KEY: ${GROQ_API_KEY:0:20}..."

# Navigate to repo root
cd "$(dirname "$0")/../../.."

echo ""
echo "📦 Installing dependencies..."
uv pip install -e . --quiet || true

echo ""
echo "🏃 Starting Banking77 task app on http://127.0.0.1:8102"
echo "Press Ctrl+C to stop"
echo ""

# Run the task app
python -m examples.task_apps.banking77.banking77_task_app \
    --host 0.0.0.0 \
    --port 8102 \
    --reload


