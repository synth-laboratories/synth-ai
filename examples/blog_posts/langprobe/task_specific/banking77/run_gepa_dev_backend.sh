#!/bin/bash
# Run GEPA in-process script against dev backend
#
# This script sets up the environment and runs the GEPA in-process script
# against the dev backend at https://synth-backend-dev-docker.onrender.com
#
# Usage:
#   ./run_gepa_dev_backend.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Setting up GEPA in-process run against dev backend...${NC}"
echo ""

# Find synth-ai repo root (go up 6 levels from script)
SYNTH_AI_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
ENV_FILE="$SYNTH_AI_ROOT/.env"

echo "Synth-ai root: $SYNTH_AI_ROOT"
echo "Env file: $ENV_FILE"
echo ""

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Warning: .env file not found at $ENV_FILE${NC}"
    echo "Creating .env file template..."
    touch "$ENV_FILE"
    echo "# Add your API keys here" >> "$ENV_FILE"
    echo "GROQ_API_KEY=" >> "$ENV_FILE"
    echo "SYNTH_API_KEY=" >> "$ENV_FILE"
    echo "ENVIRONMENT_API_KEY=" >> "$ENV_FILE"
    echo ""
    echo -e "${YELLOW}Please edit $ENV_FILE and add your API keys${NC}"
    exit 1
fi

# Source the .env file
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE..."
    set -a
    source "$ENV_FILE"
    set +a
fi

# Check for required environment variables
MISSING_VARS=()

if [ -z "$GROQ_API_KEY" ]; then
    MISSING_VARS+=("GROQ_API_KEY")
fi

if [ -z "$SYNTH_API_KEY" ]; then
    echo -e "${YELLOW}Warning: SYNTH_API_KEY not set, will default to 'test'${NC}"
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo -e "${YELLOW}Warning: ENVIRONMENT_API_KEY not set, will default to 'test'${NC}"
fi

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please set these in $ENV_FILE:"
    echo "  GROQ_API_KEY=your-groq-key"
    echo "  SYNTH_API_KEY=your-synth-key  # optional, defaults to 'test'"
    echo "  ENVIRONMENT_API_KEY=your-env-key  # optional, defaults to 'test'"
    exit 1
fi

echo -e "${GREEN}✓ Required environment variables are set${NC}"
echo ""

# Set backend URL to dev backend
export BACKEND_BASE_URL="https://synth-backend-dev-docker.onrender.com"

echo -e "${GREEN}Configuration:${NC}"
echo "  Backend: $BACKEND_BASE_URL"
echo "  GROQ_API_KEY: ${GROQ_API_KEY:0:10}..."
echo "  SYNTH_API_KEY: ${SYNTH_API_KEY:-test (default)}"
echo "  ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:-test (default)}"
echo ""

# Verify the script exists
if [ ! -f "run_gepa_in_process.py" ]; then
    echo -e "${RED}Error: run_gepa_in_process.py not found in current directory${NC}"
    exit 1
fi

# Verify config file exists
if [ ! -f "banking77_gepa.toml" ]; then
    echo -e "${RED}Error: banking77_gepa.toml not found in current directory${NC}"
    exit 1
fi

echo -e "${GREEN}Running GEPA in-process script...${NC}"
echo ""

# Run the script
uv run python run_gepa_in_process.py

