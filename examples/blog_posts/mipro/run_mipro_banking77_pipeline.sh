#!/bin/bash
# Run MIPROv2 optimisation for the multi-step Banking77 pipeline against the backend

set -euo pipefail

echo "🔬 Running MIPROv2 on Banking77 Pipeline"
echo "========================================"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

SAVED_BACKEND_BASE_URL="${BACKEND_BASE_URL:-}"
SAVED_SYNTH_BASE_URL="${SYNTH_BASE_URL:-}"

_load_env_file() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        echo "📝 Loading environment variables from $env_file..."
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            # Only process lines with '=' character
            [[ ! "$line" =~ = ]] && continue
            
            if [[ "$line" =~ ^[[:space:]]*BACKEND_BASE_URL= ]]; then
                if [ -z "$SAVED_BACKEND_BASE_URL" ]; then
                    export "$line" 2>/dev/null || true
                else
                    echo "   ⚠️  Skipping BACKEND_BASE_URL from .env (using environment value)"
                fi
            elif [[ "$line" =~ ^[[:space:]]*SYNTH_BASE_URL= ]]; then
                if [ -z "$SAVED_SYNTH_BASE_URL" ]; then
                    export "$line" 2>/dev/null || true
                else
                    echo "   ⚠️  Skipping SYNTH_BASE_URL from .env (using environment value)"
                fi
            elif [[ "$line" =~ ^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*= ]]; then
                export "$line" 2>/dev/null || true
            fi
        done < "$env_file"
    fi
}

_load_env_file "$REPO_ROOT/.env"
_load_env_file "$REPO_ROOT/examples/rl/.env"

if [ -n "$SAVED_BACKEND_BASE_URL" ]; then
    export BACKEND_BASE_URL="$SAVED_BACKEND_BASE_URL"
    echo "✅ Using BACKEND_BASE_URL from environment: $BACKEND_BASE_URL"
fi
if [ -n "$SAVED_SYNTH_BASE_URL" ]; then
    export SYNTH_BASE_URL="$SAVED_SYNTH_BASE_URL"
    echo "✅ Using SYNTH_BASE_URL from environment: $SYNTH_BASE_URL"
fi

if [ -z "${SYNTH_API_KEY:-}" ]; then
    echo "❌ Error: SYNTH_API_KEY not set"
    exit 1
fi

if [ -z "${ENVIRONMENT_API_KEY:-}" ]; then
    echo "❌ Error: ENVIRONMENT_API_KEY not set"
    exit 1
fi

if [ -z "${GROQ_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "⚠️  Warning: Neither GROQ_API_KEY nor OPENAI_API_KEY is set."
    echo "The policy defaults to Groq-hosted OSS models. Set GROQ_API_KEY to avoid failures."
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set (required for meta-model gpt-4o-mini)."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ -z "${BACKEND_BASE_URL:-}" ]; then
    BACKEND_BASE_URL="http://localhost:8000"
fi
BACKEND_URL="$BACKEND_BASE_URL"
BACKEND_URL_NO_API="${BACKEND_URL%/api}"

echo ""
echo "🔧 Debug Info:"
echo "   BACKEND_BASE_URL current: $BACKEND_BASE_URL"
echo "   BACKEND_URL: $BACKEND_URL"
echo ""

echo "✅ SYNTH_API_KEY: ${SYNTH_API_KEY:0:20}..."
echo "✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
if [ -n "${GROQ_API_KEY:-}" ]; then
    echo "✅ GROQ_API_KEY: ${GROQ_API_KEY:0:20}..."
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
fi
echo "✅ Backend URL: $BACKEND_URL"
echo ""

CONFIG_PATH="examples/blog_posts/mipro/configs/banking77_pipeline_mipro_local.toml"

# ALWAYS read task_app_url from TOML (never use any pre-existing shell variable)
# Override only if TASK_APP_URL_OVERRIDE is explicitly set
if [ -n "${TASK_APP_URL_OVERRIDE:-}" ]; then
    TASK_APP_URL="$TASK_APP_URL_OVERRIDE"
    echo "📝 Using OVERRIDE task app URL: $TASK_APP_URL"
else
    # Extract task_app_url from TOML using grep/sed (no Python dependencies)
    # Pattern: task_app_url = "https://..."
    TASK_APP_URL="$(grep "^task_app_url" "$CONFIG_PATH" | sed -E 's/^[^=]*=[[:space:]]*"([^"]*)".*/\1/' | head -1)"
    if [ -z "$TASK_APP_URL" ]; then
        echo "❌ ERROR: task_app_url not found in $CONFIG_PATH" >&2
        echo "   Please ensure the config file contains: task_app_url = \"...\"" >&2
        exit 1
    fi
    echo "📝 Task app URL from TOML: $TASK_APP_URL"
fi
echo ""

echo "🔍 Checking if Banking77 pipeline task app is running on ${TASK_APP_URL}..."
if ! curl -s -f -H "X-API-Key: $ENVIRONMENT_API_KEY" "$TASK_APP_URL/health" > /dev/null 2>&1; then
    cat <<EOF
❌ Error: Banking77 pipeline task app is not running on ${TASK_APP_URL}

Start it with:
  uvx synth-ai deploy banking77-pipeline --runtime uvicorn --port 8112 --env-file .env --follow
  # or deploy to Modal dev: modal deploy --env dev examples/task_apps/banking77_pipeline/deploy_wrapper.py
EOF
    exit 1
fi
echo "✅ Pipeline task app is healthy"
echo ""

echo "🔍 Checking backend connection to $BACKEND_URL..."
if ! curl -s -f "$BACKEND_URL_NO_API/api/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: Cannot connect to backend at $BACKEND_URL"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Backend is healthy"
fi
echo ""

echo "🚀 Starting MIPROv2 training..."
echo "Config: $CONFIG_PATH"
echo ""
echo "Multi-Step Flow:"
echo "  1. Bootstrap: two-module pipeline on seeds [0-14]"
echo "  2. Optimisation: 5 iterations × 2 variants (classifier + calibrator each evaluation)"
echo "  3. Held-out evaluation on seeds [40-49]"
echo ""

export BACKEND_BASE_URL="$BACKEND_URL"
export SYNTH_BASE_URL="$BACKEND_URL"

uvx synth-ai train \
    --type prompt_learning \
    --config "$CONFIG_PATH" \
    --backend "$BACKEND_URL" \
    --poll

echo ""
echo "✅ MIPROv2 pipeline training complete!"

