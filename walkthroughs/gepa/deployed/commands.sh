#!/bin/bash
# Interactive GEPA Deployment Script
# This script guides you through deploying Banking77 task app and running GEPA optimization

# Note: We don't use set -e because some commands (like pkill) are expected to fail

# Get script directory and repo root
# Script is at: walkthroughs/gepa/deployed/commands.sh
# Need to go up 3 levels: deployed -> gepa -> walkthroughs -> synth-ai (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to repo root so all paths work correctly
cd "$REPO_ROOT"

echo "Changed to repo root: $(pwd)"
echo "Verifying task app path exists..."
if [ ! -f "examples/task_apps/banking77/banking77_task_app.py" ]; then
    echo -e "${YELLOW}⚠️  Warning: Task app not found at expected path${NC}"
    echo "Looking for: examples/task_apps/banking77/banking77_task_app.py"
    echo "Current directory: $(pwd)"
    exit 1
fi
echo -e "${GREEN}✓ Task app found${NC}"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to prompt user
prompt_step() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "$2"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel... "
    echo ""
}

# Function to execute command with output
run_command() {
    echo -e "${YELLOW}Executing:${NC} $1"
    echo ""
    eval "$1"
    echo ""
}

# Check prerequisites
echo -e "${GREEN}GEPA Deployment Walkthrough${NC}"
echo "================================"
echo ""
echo "Checking prerequisites..."

if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${YELLOW}Warning: GROQ_API_KEY not set in environment${NC}"
fi

if [ -z "$SYNTH_API_KEY" ]; then
    echo -e "${YELLOW}Warning: SYNTH_API_KEY not set in environment${NC}"
    echo "Make sure to set it in your .env file or export it before running this script"
fi

# Ensure working directory
mkdir -p /tmp/gepa_walkthrough/results

# Step 1: Generate ENVIRONMENT_API_KEY
prompt_step "Step 1: Generate ENVIRONMENT_API_KEY" \
"This step creates a new API key for authenticating with the task app and registers it with the backend.
The key will be stored in /tmp/gepa_walkthrough/cli_env.txt"

echo "Generating ENVIRONMENT_API_KEY..."
ENV_KEY=$(uv run python -c "from synth_ai.learning.rl.secrets import mint_environment_api_key; print(mint_environment_api_key())" 2>&1 | tail -1 | tr -d '\n' | tr -d '\r')
echo "ENVIRONMENT_API_KEY=$ENV_KEY" > /tmp/gepa_walkthrough/cli_env.txt
echo "TASK_APP_URL=" >> /tmp/gepa_walkthrough/cli_env.txt

echo -e "${GREEN}✓ ENVIRONMENT_API_KEY generated${NC}"
echo "Key: ${ENV_KEY:0:20}..."

# Register key with backend
echo ""
echo "Registering key with backend..."
run_command 'uv run python -c "
import os
from pathlib import Path
from synth_ai.learning.rl.secrets import mint_environment_api_key
from synth_ai.cli.lib.task_app_env import preflight_env_key

env_file = Path(\"/tmp/gepa_walkthrough/cli_env.txt\")
try:
    preflight_env_key([env_file], crash_on_failure=False)
    print(\"✅ Key registered with backend\")
except Exception as e:
    print(f\"⚠️  Registration warning: {e}\")
    print(\"   Continuing anyway...\")
"'

# Step 2: Deploy tunnel
prompt_step "Step 2: Deploy Cloudflare Tunnel" \
"This step:
- Kills any existing processes on port 8102
- Starts the Banking77 task app locally
- Creates a Cloudflare tunnel to expose it publicly
- Writes TASK_APP_URL to the env file when ready

The deploy command runs in the background. We'll wait a few seconds for it to start."

run_command 'pkill -f "cloudflared.*8102" 2>/dev/null || true'
run_command 'pkill -f "uvicorn.*8102" 2>/dev/null || true'
run_command 'lsof -ti :8102 2>/dev/null | xargs kill -9 2>/dev/null || true'
run_command 'sleep 2'

echo "Starting tunnel deployment in background..."
TASK_APP_PATH="$REPO_ROOT/examples/task_apps/banking77/banking77_task_app.py"
echo "Task app path: $TASK_APP_PATH"
if [ ! -f "$TASK_APP_PATH" ]; then
    echo -e "${YELLOW}⚠️  Error: Task app not found at $TASK_APP_PATH${NC}"
    exit 1
fi
run_command "uv run synth-ai deploy tunnel \"$TASK_APP_PATH\" --tunnel-mode quick --port 8102 --env /tmp/gepa_walkthrough/cli_env.txt &"

echo ""
echo "Waiting for tunnel to establish..."
run_command 'sleep 25'

# Step 3: Extract tunnel URL
prompt_step "Step 3: Extract Tunnel URL" \
"This step reads the TASK_APP_URL that was written by the deploy command."

echo "Extracting tunnel URL from env file..."
TASK_URL=$(grep "^TASK_APP_URL=" /tmp/gepa_walkthrough/cli_env.txt | cut -d"=" -f2- | tr -d '"' | tr -d "'" | tr -d '\n' | tr -d '\r')

if [ -z "$TASK_URL" ] || [ "$TASK_URL" = "" ]; then
    echo -e "${YELLOW}⚠️  Warning: TASK_APP_URL not found in env file${NC}"
    echo "The deploy command may still be starting. Checking again in 5 seconds..."
    sleep 5
    TASK_URL=$(grep "^TASK_APP_URL=" /tmp/gepa_walkthrough/cli_env.txt | cut -d"=" -f2- | tr -d '"' | tr -d "'" | tr -d '\n' | tr -d '\r')
fi

if [ -z "$TASK_URL" ] || [ "$TASK_URL" = "" ]; then
    echo -e "${YELLOW}⚠️  TASK_APP_URL still not found. You may need to check the deploy output manually.${NC}"
    echo "You can check the env file: cat /tmp/gepa_walkthrough/cli_env.txt"
    echo "Continuing anyway..."
else
    echo -e "${GREEN}✓ Tunnel URL extracted: $TASK_URL${NC}"
fi

# Step 4: Create GEPA config
prompt_step "Step 4: Create GEPA Config" \
"This step updates the base TOML config to:
- Use the tunnel URL for task_app_url
- Set rollout budget to 2000 (sufficient for prompt improvement)

The config will be saved to /tmp/gepa_walkthrough/banking77_gepa_prod.toml"

echo "Creating GEPA config..."
CONFIG_SOURCE="$REPO_ROOT/examples/blog_posts/langprobe/task_specific/banking77/banking77_gepa.toml"
if [ ! -f "$CONFIG_SOURCE" ]; then
    echo -e "${YELLOW}⚠️  Error: Config source not found at $CONFIG_SOURCE${NC}"
    exit 1
fi

if [ -z "$TASK_URL" ] || [ "$TASK_URL" = "" ]; then
    echo -e "${YELLOW}⚠️  No tunnel URL available. Please check the deploy output and update the config manually.${NC}"
    echo "Config template created at: /tmp/gepa_walkthrough/banking77_gepa_prod.toml"
    cat "$CONFIG_SOURCE" | sed "s|budget = .*|budget = 2000|" > /tmp/gepa_walkthrough/banking77_gepa_prod.toml
else
    cat "$CONFIG_SOURCE" | sed "s|task_app_url = \".*\"|task_app_url = \"$TASK_URL\"|" | sed "s|budget = .*|budget = 2000|" > /tmp/gepa_walkthrough/banking77_gepa_prod.toml
    echo -e "${GREEN}✓ Config created: /tmp/gepa_walkthrough/banking77_gepa_prod.toml${NC}"
fi

# Step 5: Run training
prompt_step "Step 5: Run GEPA Training" \
"This step submits the GEPA optimization job to the production backend.
The job will:
- Run 5 generations with 4 children per generation (20 prompt candidates)
- Use a rollout budget of 2000
- Stream results as it progresses

This may take several minutes. The script will poll for completion."

echo "Submitting job..."
run_command 'export BACKEND_BASE_URL="https://agent-learning.onrender.com"'
run_command 'uv run synth-ai train /tmp/gepa_walkthrough/banking77_gepa_prod.toml --backend "$BACKEND_BASE_URL" --env /tmp/gepa_walkthrough/cli_env.txt --poll'

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results are saved to: /tmp/gepa_walkthrough/results/"
echo ""
echo "To retrieve the optimized prompt, use the job ID from the output above with:"
echo "  python3 << 'PYTHON'"
echo "  import asyncio"
echo "  from synth_ai.learning.prompt_learning_client import PromptLearningClient"
echo "  from synth_ai.api.train.utils import ensure_api_base"
echo "  import os"
echo ""
echo "  async def get_results():"
echo "      job_id = 'YOUR_JOB_ID_HERE'"
echo "      backend_url = ensure_api_base('https://agent-learning.onrender.com')"
echo "      api_key = os.getenv('SYNTH_API_KEY')"
echo "      client = PromptLearningClient(backend_url, api_key)"
echo "      prompts = await client.get_prompts(job_id)"
echo "      # ... (see walkthroughs/gepa/in_process/run.py for full example)"
echo ""
echo "  asyncio.run(get_results())"
echo "PYTHON"

