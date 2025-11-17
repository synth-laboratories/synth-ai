#!/bin/bash
# Deploy the Banking77 pipeline task app locally via uvicorn

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PORT="8112"

echo "🚀 Launching banking77-pipeline task app on port ${PORT}"
uvx synth-ai deploy banking77-pipeline \
  --runtime uvicorn \
  --port "${PORT}" \
  --env-file .env \
  --follow
#!/bin/bash
# Deploy the Banking77 multi-step pipeline task app locally for MIPROv2 optimisation

set -e

echo "🚀 Deploying Banking77 Pipeline Task App for MIPROv2..."
echo "======================================================"

# Ensure ENVIRONMENT_API_KEY exists (shared with backend + CLI)
export ENVIRONMENT_API_KEY="${ENVIRONMENT_API_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(32))')}"

# Optional providers (helpful during local experimentation)
if [ -z "$GROQ_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Neither GROQ_API_KEY nor OPENAI_API_KEY is set."
    echo "   The task app can still start, but hosted inference may fail."
else
    if [ -n "$GROQ_API_KEY" ]; then
        echo "✅ GROQ_API_KEY: ${GROQ_API_KEY:0:20}..."
    fi
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
    fi
fi

echo "✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."

# Navigate to repo root
cd "$(dirname "$0")/../../.."

echo ""
echo "📦 Ensuring dependencies are installed..."
uv pip install -e . --quiet || true

echo ""
echo "🏃 Starting Banking77 pipeline task app on http://127.0.0.1:8112"
echo "Press Ctrl+C to stop."
echo ""

python -m examples.task_apps.banking77_pipeline.banking77_pipeline_task_app \
    --host 0.0.0.0 \
    --port 8112 \
    --reload

