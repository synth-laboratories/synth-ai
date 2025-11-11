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

# Navigate to repo root
cd "$(dirname "$0")/../../.."

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

TASK_APP_PATH="examples/task_apps/banking77/banking77_task_app.py"
ENV_FILE=".env.tunnel"

# Deploy with quick tunnel (free, ephemeral)
uvx synth-ai deploy \
    --task-app "$TASK_APP_PATH" \
    --runtime tunnel \
    --tunnel-mode quick \
    --port 8102 \
    --env "$ENV_FILE" \
    --trace

# Read tunnel URL from .env
if [ -f "$ENV_FILE" ]; then
    TASK_APP_URL=$(grep "^TASK_APP_URL=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    if [ -z "$TASK_APP_URL" ]; then
        echo "❌ ERROR: TASK_APP_URL not found in $ENV_FILE"
        exit 1
    fi
    echo ""
    echo "✅ Tunnel deployed: $TASK_APP_URL"
    echo "   Credentials saved to: $ENV_FILE"
else
    echo "❌ ERROR: $ENV_FILE not created"
    exit 1
fi

# Verify tunnel is accessible
echo ""
echo "🔍 Verifying tunnel health..."
if curl -s -f -H "X-API-Key: $ENVIRONMENT_API_KEY" "$TASK_APP_URL/health" > /dev/null 2>&1; then
    echo "✅ Tunnel is healthy"
else
    echo "⚠️  Warning: Tunnel health check failed, but continuing..."
fi

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
CONFIG_DIR="examples/tunnel_gepa_banking77"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/banking77_gepa_tunnel.toml"

echo ""
echo "📝 Creating GEPA config: $CONFIG_FILE"
cat > "$CONFIG_FILE" <<EOF
# GEPA Prompt Learning for Banking77 via Cloudflare Tunnel
# This config uses a Cloudflare Tunnel URL to expose the local task app
# to Synth's production backend for prompt optimization.

[prompt_learning]
algorithm = "gepa"
task_app_url = "$TASK_APP_URL"
task_app_id = "banking77"

# Initial prompt pattern
[prompt_learning.initial_prompt]
id = "banking77_classifier"
name = "Banking77 Intent Classification"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are an expert banking assistant. \n\n**Available Banking Intents:**\n{available_intents}\n\n**Task:**\nCall the \`banking77_classify\` tool with the \`intent\` parameter set to ONE of the intent labels listed above that best matches the customer query. The intent must be an exact match from the list."
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
order = 1

[prompt_learning.initial_prompt.wildcards]
query = "REQUIRED"
available_intents = "OPTIONAL"

# Policy configuration
[prompt_learning.policy]
inference_mode = "synth_hosted"
model = "openai/gpt-oss-20b"
provider = "groq"
temperature = 0.0
max_completion_tokens = 512
policy_name = "banking77-classifier"

# Training split config
[prompt_learning.env_config]
pool = "train"

# GEPA-specific configuration
[prompt_learning.gepa]
env_name = "banking77"
proposer_type = "dspy"

# Rollout configuration
[prompt_learning.gepa.rollout]
budget = 100
max_concurrent = 20
minibatch_size = 10

# Evaluation configuration
[prompt_learning.gepa.evaluation]
seeds = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
validation_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
validation_pool = "validation"
validation_top_k = 3
test_pool = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

# Mutation configuration
[prompt_learning.gepa.mutation]
rate = 0.3
llm_model = "openai/gpt-oss-120b"
llm_provider = "groq"
llm_inference_url = "https://api.groq.com/openai/v1"

# Population configuration
[prompt_learning.gepa.population]
initial_size = 10
num_generations = 3
children_per_generation = 12
crossover_rate = 0.5
selection_pressure = 1.0
patience_generations = 3

# Archive configuration
[prompt_learning.gepa.archive]
size = 40
pareto_set_size = 32
pareto_eps = 1e-6
feedback_fraction = 0.5

# Token configuration
[prompt_learning.gepa.token]
counting_model = "gpt-4"
enforce_pattern_limit = true
EOF

echo "✅ Config created"
echo ""

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

uvx synth-ai train \
    --type prompt_learning \
    --config "$CONFIG_FILE" \
    --backend "$BACKEND_URL" \
    --poll

echo ""
echo "✅ GEPA optimization complete!"
echo ""
echo "📊 Results:"
echo "   - Config: $CONFIG_FILE"
echo "   - Tunnel URL: $TASK_APP_URL"
echo "   - Credentials: $ENV_FILE"
echo ""
echo "💡 To view results, check the job status in the Synth dashboard:"
echo "   https://app.usesynth.ai"

