#!/bin/bash
# Deploy Banking77 task app via Cloudflare Tunnel and run GEPA optimization
# This example demonstrates using Cloudflare Tunnel to expose a local task app
# to Synth's production backend for prompt optimization.

set -e

echo "🚀 Banking77 GEPA Optimization via Cloudflare Tunnel"
echo "====================================================="
echo ""

# Load .env file if it exists
ENV_FILES=(".env" "$(dirname "$0")/../../../.env" "$HOME/.synth-ai/.env")
for env_file in "${ENV_FILES[@]}"; do
    if [ -f "$env_file" ]; then
        echo "📝 Loading environment from: $env_file"
        set -a
        source "$env_file"
        set +a
        break
    fi
done

# Check required environment variables
if [ -z "$SYNTH_API_KEY" ]; then
    echo "❌ ERROR: SYNTH_API_KEY not set"
    echo "   Get your API key from: https://app.usesynth.ai/api-keys"
    exit 1
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "⚠️  ENVIRONMENT_API_KEY not set, generating one..."
    export ENVIRONMENT_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    echo "✅ Generated ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  Warning: GROQ_API_KEY not set (needed for LLM-guided mutations in GEPA)"
    echo "   Set it with: export GROQ_API_KEY=your_key"
fi

# Use production backend
BACKEND_URL="${BACKEND_BASE_URL:-https://api.usesynth.ai}"
echo "📍 Backend URL: $BACKEND_URL"
echo "✅ SYNTH_API_KEY: ${SYNTH_API_KEY:0:20}..."
echo "✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
echo ""

# Navigate to repo root (script is at examples/tunnel_gepa_banking77/run_gepa_with_tunnel.sh)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ ERROR: cloudflared not found"
    echo "   Install it:"
    echo "     macOS: brew install cloudflare/cloudflare/cloudflared"
    echo "     Linux/Windows: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/"
    exit 1
fi

echo "✅ cloudflared found: $(which cloudflared)"
echo ""

# Deploy task app via Cloudflare Tunnel
echo "🌐 Deploying Banking77 task app via Cloudflare Tunnel..."
echo ""

TASK_APP_PATH="$REPO_ROOT/examples/task_apps/banking77/banking77_task_app.py"
ENV_FILE="$REPO_ROOT/.env.tunnel"

if [ ! -f "$TASK_APP_PATH" ]; then
    echo "❌ ERROR: Task app not found: $TASK_APP_PATH"
    exit 1
fi
echo "✅ Task app found: $TASK_APP_PATH"
echo ""

# Create .env file if it doesn't exist (deploy command requires it to exist)
touch "$ENV_FILE"

# Deploy with quick tunnel (free, ephemeral) in background mode
# The deploy command returns immediately, keeping tunnel running headlessly
echo "🚀 Starting tunnel deployment (background mode)..."
uv run synth-ai deploy \
    --task-app "$TASK_APP_PATH" \
    --runtime tunnel \
    --tunnel-mode quick \
    --port 8102 \
    --env "$ENV_FILE" \
    --trace > /tmp/tunnel_deploy.log 2>&1 &
DEPLOY_PID=$!

# Wait for tunnel URL to be written to .env file
echo "⏳ Waiting for tunnel to be ready..."
TASK_APP_URL=""
for i in {1..30}; do
    if [ -f "$ENV_FILE" ] && grep -q "^TASK_APP_URL=" "$ENV_FILE"; then
        TASK_APP_URL=$(grep "^TASK_APP_URL=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [ -n "$TASK_APP_URL" ]; then
            echo "✅ Tunnel URL found: $TASK_APP_URL"
            # Wait a bit more for DNS propagation and tunnel to be fully ready
            echo "⏳ Waiting for tunnel to be accessible..."
            for j in {1..15}; do
                if curl -s -f -H "X-API-Key: $ENVIRONMENT_API_KEY" "$TASK_APP_URL/health" > /dev/null 2>&1; then
                    echo "✅ Tunnel is accessible!"
                    break
                fi
                if [ $j -eq 15 ]; then
                    echo "⚠️  Tunnel not yet accessible, but continuing..."
                fi
                sleep 2
            done
            break
        fi
    fi
    sleep 1
done

if [ -z "$TASK_APP_URL" ]; then
    echo "❌ ERROR: Tunnel deployment failed or timed out"
    echo "   Check deployment logs: /tmp/tunnel_deploy.log"
    kill $DEPLOY_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "✅ Tunnel deployed: $TASK_APP_URL"
echo "   Credentials saved to: $ENV_FILE"
echo "   Tunnel process PID: $DEPLOY_PID (running in background)"

# Verify backend is accessible
echo ""
echo "🔍 Verifying backend connection..."
if curl -s -f "$BACKEND_URL/api/v1/health" > /dev/null 2>&1; then
    echo "✅ Backend is accessible"
else
    echo "⚠️  Warning: Cannot connect to backend at $BACKEND_URL"
    echo "   Continuing anyway..."
fi

# Create GEPA config with tunnel URL
CONFIG_DIR="$REPO_ROOT/examples/tunnel_gepa_banking77"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/banking77_gepa_tunnel.toml"

echo ""
echo "📝 Creating GEPA config: $CONFIG_FILE"
# Copy the base config from existing Banking77 GEPA example and update task_app_url
BASE_CONFIG="$REPO_ROOT/examples/blog_posts/gepa/configs/banking77_gepa_local.toml"
if [ -f "$BASE_CONFIG" ]; then
    # Copy base config and update task_app_url
    cp "$BASE_CONFIG" "$CONFIG_FILE"
    # Update task_app_url using sed (works on macOS and Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|task_app_url = \".*\"|task_app_url = \"$TASK_APP_URL\"|" "$CONFIG_FILE"
    else
        sed -i "s|task_app_url = \".*\"|task_app_url = \"$TASK_APP_URL\"|" "$CONFIG_FILE"
    fi
    echo "✅ Config created from: $BASE_CONFIG"
    echo "   Updated task_app_url to: $TASK_APP_URL"
else
    echo "❌ ERROR: Base config not found: $BASE_CONFIG"
    echo "   Please ensure the Banking77 GEPA example config exists"
    exit 1
fi

# Run GEPA optimization
echo "🎯 Starting GEPA prompt optimization..."
echo "   Config: $CONFIG_FILE"
echo "   Backend: $BACKEND_URL"
echo "   Task App: $TASK_APP_URL"
echo ""
echo "⚠️  Note: Keep the tunnel process running in another terminal"
echo "   The tunnel will close when you stop the deployment process"
echo ""

export BACKEND_BASE_URL="$BACKEND_URL"
export SYNTH_BASE_URL="$BACKEND_URL"

# Run GEPA optimization
# The tunnel is already running (uvicorn in background thread, cloudflared process)
# We'll run GEPA training which will submit the job to the backend
echo ""
echo "🚀 Starting GEPA prompt optimization..."
echo "   Config: $CONFIG_FILE"
echo "   Backend: $BACKEND_URL"
echo "   Task App: $TASK_APP_URL"
echo ""
echo "⚠️  Note: The tunnel process is running in the background."
echo "   Keep this terminal open until training completes."
echo ""

export BACKEND_BASE_URL="$BACKEND_URL"
export SYNTH_BASE_URL="$BACKEND_URL"

# Run GEPA training
# Use --env-file to skip the interactive prompt
uv run synth-ai train \
    --type prompt_learning \
    --config "$CONFIG_FILE" \
    --backend "$BACKEND_URL" \
    --env-file "$ENV_FILE" \
    --poll

echo ""
echo "✅ GEPA optimization complete!"
echo ""

# Cleanup: stop tunnel processes
echo "🧹 Cleaning up tunnel processes..."
kill $DEPLOY_PID 2>/dev/null || true
pkill -f "cloudflared.*8102" 2>/dev/null || true
pkill -f "uvicorn.*8102" 2>/dev/null || true
sleep 1

echo ""
echo "📊 Results:"
echo "   - Config: $CONFIG_FILE"
echo "   - Tunnel URL: $TASK_APP_URL"
echo "   - Credentials: $ENV_FILE"
echo ""
echo "💡 To view results, check the job status in the Synth dashboard:"
echo "   https://app.usesynth.ai"

