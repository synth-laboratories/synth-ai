#!/bin/bash
# Run MIPROv2 optimization for Banking77 against the backend

set -e

echo "🔬 Running MIPROv2 on Banking77"
echo "================================="

# Navigate to repo root
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# Save backend-related vars from environment before loading .env files (so they don't get overridden)
SAVED_BACKEND_BASE_URL="${BACKEND_BASE_URL:-}"
SAVED_SYNTH_BASE_URL="${SYNTH_BASE_URL:-}"

# Load environment variables from .env file if it exists
# Use a safer method that only loads KEY=VALUE pairs and ignores errors
_load_env_file() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        echo "📝 Loading environment variables from $env_file..."
        # Only export lines that look like KEY=VALUE (handles comments and empty lines)
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            # Only export if it looks like KEY=VALUE
            # BUT: Don't override backend URLs if they were set in environment
            if [[ "$line" =~ ^[[:space:]]*BACKEND_BASE_URL= ]]; then
                if [ -z "$SAVED_BACKEND_BASE_URL" ]; then
                    # Only load from .env if not already set in environment
                    export "$line" 2>/dev/null || true
                else
                    echo "   ⚠️  Skipping BACKEND_BASE_URL from .env (using environment value)"
                fi
            elif [[ "$line" =~ ^[[:space:]]*SYNTH_BASE_URL= ]]; then
                if [ -z "$SAVED_SYNTH_BASE_URL" ]; then
                    # Only load from .env if not already set in environment
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

# Restore backend URLs from environment if they were set
if [ -n "$SAVED_BACKEND_BASE_URL" ]; then
    export BACKEND_BASE_URL="$SAVED_BACKEND_BASE_URL"
    echo "✅ Using BACKEND_BASE_URL from environment: $BACKEND_BASE_URL"
fi
if [ -n "$SAVED_SYNTH_BASE_URL" ]; then
    export SYNTH_BASE_URL="$SAVED_SYNTH_BASE_URL"
    echo "✅ Using SYNTH_BASE_URL from environment: $SYNTH_BASE_URL"
fi

# Check for required environment variables
if [ -z "$SYNTH_API_KEY" ]; then
    echo "❌ Error: SYNTH_API_KEY not set"
    echo "Please get your API key from the backend and set it:"
    echo "  export SYNTH_API_KEY=your_key"
    echo "Or add it to $REPO_ROOT/.env"
    exit 1
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "❌ Error: ENVIRONMENT_API_KEY not set"
    echo "Please set the same key used when deploying the task app:"
    echo "  export ENVIRONMENT_API_KEY=your_key"
    echo "Or add it to $REPO_ROOT/.env"
    exit 1
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY not set"
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY=your_key"
    echo "Or add it to $REPO_ROOT/.env"
    exit 1
fi

# Check for OpenAI API key (needed for meta-model)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "MIPROv2 uses a meta-model (gpt-4o-mini) for prompt proposals."
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY=your_key"
    echo "Or add it to $REPO_ROOT/.env"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default to localhost backend if not specified
# Respect BACKEND_BASE_URL from environment (don't override if already set)
if [ -z "$BACKEND_BASE_URL" ]; then
    BACKEND_BASE_URL="http://localhost:8000"
fi
BACKEND_URL="$BACKEND_BASE_URL"
# Ensure it doesn't have /api suffix for the base URL check (CLI will add it)
BACKEND_URL_NO_API="${BACKEND_URL%/api}"

echo ""
echo "🔧 Debug Info:"
echo "   BACKEND_BASE_URL from env: ${SAVED_BACKEND_BASE_URL:-<not set>}"
echo "   BACKEND_BASE_URL current: $BACKEND_BASE_URL"
echo "   BACKEND_URL: $BACKEND_URL"
echo ""

echo "✅ SYNTH_API_KEY: ${SYNTH_API_KEY:0:20}..."
echo "✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
echo "✅ GROQ_API_KEY: ${GROQ_API_KEY:0:20}..."
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
fi
echo "✅ Backend URL: $BACKEND_URL"
echo ""

# Already navigated to repo root above

# Check if task app is running
echo "🔍 Checking if Banking77 task app is running on http://127.0.0.1:8102..."
if ! curl -s -f -H "X-API-Key: $ENVIRONMENT_API_KEY" http://127.0.0.1:8102/health > /dev/null 2>&1; then
    echo "❌ Error: Banking77 task app is not running on http://127.0.0.1:8102"
    echo ""
    echo "Please start it first:"
    echo "  ./examples/blog_posts/mipro/deploy_banking77_task_app.sh"
    echo ""
    echo "Or in another terminal:"
    echo "  cd $(pwd)"
    echo "  uvx synth-ai deploy banking77 --runtime uvicorn --port 8102"
    exit 1
fi
echo "✅ Task app is healthy"
echo ""

# Check backend connection
echo "🔍 Checking backend connection to $BACKEND_URL..."
if ! curl -s -f "$BACKEND_URL_NO_API/api/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: Cannot connect to backend at $BACKEND_URL"
    echo "Make sure the backend is running."
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
echo "Config: examples/blog_posts/mipro/configs/banking77_mipro_local.toml"
echo ""
echo "MIPROv2 Flow:"
echo "  1. Bootstrap Phase: Evaluate baseline on seeds [0-4], collect few-shot examples"
echo "  2. Optimization Loop: 16 iterations × 6 variants = 96 evaluations"
echo "  3. Final Evaluation: Test on held-out seeds [10-19]"
echo ""

# Export backend URLs so CLI respects them (overrides .env files)
export BACKEND_BASE_URL="$BACKEND_URL"
# Also set SYNTH_BASE_URL to match (CLI may check this as fallback)
export SYNTH_BASE_URL="$BACKEND_URL"

echo "🚀 Running CLI with:"
echo "   BACKEND_BASE_URL=$BACKEND_BASE_URL"
echo "   SYNTH_BASE_URL=$SYNTH_BASE_URL"
echo "   --backend=$BACKEND_URL"
echo ""

# Run the training
uvx synth-ai train \
    --type prompt_learning \
    --config examples/blog_posts/mipro/configs/banking77_mipro_local.toml \
    --backend "$BACKEND_URL" \
    --poll

echo ""
echo "✅ MIPROv2 training complete!"

