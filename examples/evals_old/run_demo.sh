#!/bin/bash

# Run Crafter experiments comparing gpt-5-nano and Qwen/Qwen3-32B-Instruct

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the synth-ai root directory
cd "$SCRIPT_DIR/../.."

# Interactive mini-demo: run small comparison, then analyze v3 traces
set -euo pipefail

# Load env (prefer local .env at repo root)
set +u
set -a
if [ -f ".env" ]; then source ".env"; fi
set +a
set -u

# Ensure API key present (SYNTH_API_KEY, optionally mirror to OPENAI_API_KEY)
ensure_api_key() {
  local current_key="${SYNTH_API_KEY:-}"
  if [ -n "$current_key" ]; then
    local preview="${current_key:0:6}...${current_key: -4}"
    read -r -p "Detected SYNTH_API_KEY ($preview). Use this key? [Y/n]: " USE_CUR || true
    USE_CUR=${USE_CUR:-Y}
    if [[ ! "$USE_CUR" =~ ^[Yy]$ ]]; then
      current_key=""
    fi
  fi

  if [ -z "$current_key" ] && [ -n "${SYNTH_API_KEY_PROD:-}" ]; then
    local prod_prev="${SYNTH_API_KEY_PROD:0:6}...${SYNTH_API_KEY_PROD: -4}"
    read -r -p "Use SYNTH_API_KEY_PROD ($prod_prev)? [y/N]: " USE_PROD || true
    if [[ "$USE_PROD" =~ ^[Yy]$ ]]; then
      current_key="$SYNTH_API_KEY_PROD"
    fi
  fi

  while [ -z "$current_key" ]; do
    echo
    read -s -p "Enter your SYNTH_API_KEY: " KEY_IN || true
    echo
    if [ -n "$KEY_IN" ]; then
      current_key="$KEY_IN"
    else
      echo "A valid SYNTH_API_KEY is required to continue."
    fi
  done

  export SYNTH_API_KEY="$current_key"
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY="$SYNTH_API_KEY"
    echo "OPENAI_API_KEY set from SYNTH_API_KEY."
  fi
}

# Interactive prompts (with sensible defaults)
MODELS_DEFAULT="gpt-5-nano gpt-4.1-nano"
read -r -p "Models to compare (space-separated) [${MODELS_DEFAULT}]: " MODELS_INPUT || true
MODELS=${MODELS_INPUT:-$MODELS_DEFAULT}
echo "Models: ${MODELS}"

read -r -p "Episodes per model [3]: " EPISODES_INPUT || true
EPISODES=${EPISODES_INPUT:-3}

read -r -p "Max turns per episode [5]: " MAX_TURNS_INPUT || true
MAX_TURNS=${MAX_TURNS_INPUT:-5}

read -r -p "Parallelism per model (concurrency) [5]: " CONCURRENCY_INPUT || true
CONCURRENCY=${CONCURRENCY_INPUT:-5}

read -r -p "Difficulty [easy]: " DIFFICULTY_INPUT || true
DIFFICULTY=${DIFFICULTY_INPUT:-easy}

echo "Running comparison: episodes=${EPISODES}, max_turns=${MAX_TURNS}, difficulty=${DIFFICULTY}, concurrency=${CONCURRENCY}"

# Ensure key before running rollouts
ensure_api_key

uv run python examples/evals/compare_models.py \
    --episodes "${EPISODES}" \
    --max-turns "${MAX_TURNS}" \
    --difficulty "${DIFFICULTY}" \
    --models ${MODELS} \
    --base-seed 1000 \
    --turn-timeout 20.0 \
    --episode-timeout 180.0 \
    --concurrency "${CONCURRENCY}" \
    --quiet

# Derive v3 sqld internal DB path for quick analysis
DB_PATH="$PWD/traces/v3/synth_ai.db/dbs/default/data"
export DB_PATH
echo "Using v3 traces DB: $DB_PATH"

echo "\nAvailable achievements (session counts):"
uv run python -m examples.evals.trace_analysis list --db "$DB_PATH"

echo "\nEnter achievements to filter by (space-separated), or press Enter for 'collect_wood':"
read -r ACH
ACH=${ACH:-collect_wood}

echo "Optionally restrict to models (space-separated), or press Enter to include all:"
read -r MODELS_FILTER

mkdir -p ft_data
if [ -n "$MODELS_FILTER" ]; then
  echo "\nRunning: uv run python -m examples.evals.trace_analysis filter --db \"$DB_PATH\" --achievements $ACH --output ft_data/evals_filtered.jsonl --models $MODELS_FILTER"
  uv run python -m examples.evals.trace_analysis filter --db "$DB_PATH" --achievements $ACH --output ft_data/evals_filtered.jsonl --models $MODELS_FILTER
else
  echo "\nRunning: uv run python -m examples.evals.trace_analysis filter --db \"$DB_PATH\" --achievements $ACH --output ft_data/evals_filtered.jsonl"
  uv run python -m examples.evals.trace_analysis filter --db "$DB_PATH" --achievements $ACH --output ft_data/evals_filtered.jsonl
fi

# Show stats comparing filtered vs others (including achievement frequencies)
if [ -n "$MODELS_FILTER" ]; then
  echo "\nRunning: uv run python -m examples.evals.trace_analysis stats --db \"$DB_PATH\" --achievements $ACH --models $MODELS_FILTER"
  uv run python -m examples.evals.trace_analysis stats --db "$DB_PATH" --achievements $ACH --models $MODELS_FILTER
else
  echo "\nRunning: uv run python -m examples.evals.trace_analysis stats --db \"$DB_PATH\" --achievements $ACH"
  uv run python -m examples.evals.trace_analysis stats --db "$DB_PATH" --achievements $ACH
fi

echo "\nDone. See ft_data/evals_filtered.jsonl and v3 DB for deeper analysis."