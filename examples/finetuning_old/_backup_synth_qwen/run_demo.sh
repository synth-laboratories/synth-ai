#!/usr/bin/env bash

# Interactive demo for Qwen 4B Crafter finetuning
# Mirrors the flow in readme.md and example_log.md

set -euo pipefail

# Locate repo root and cd there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/../../.."

echo "Synth Qwen4B finetuning demo (Crafter)"

# Load env (prefer example-local .env, then repo .env)
set +u
set -a
if [ -f "$SCRIPT_DIR/.env" ]; then source "$SCRIPT_DIR/.env"; fi
if [ -f ".env" ]; then source ".env"; fi
set +a
set -u

# Helper: prompt with default
prompt() {
  local msg="$1"; shift
  local default="$1"; shift
  local var
  read -r -p "$msg" var || true
  if [ -z "$var" ]; then
    echo "$default"
  else
    echo "$var"
  fi
}

# Ensure API key present (and set OPENAI_API_KEY fallback)
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

  if [ -z "$current_key" ]; then
    if [ -n "${SYNTH_API_KEY_PROD:-}" ]; then
      local prod_prev="${SYNTH_API_KEY_PROD:0:6}...${SYNTH_API_KEY_PROD: -4}"
      read -r -p "Use SYNTH_API_KEY_PROD ($prod_prev)? [y/N]: " USE_PROD || true
      if [[ "$USE_PROD" =~ ^[Yy]$ ]]; then
        current_key="$SYNTH_API_KEY_PROD"
      fi
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

# Step 1: Rollouts to generate v3 traces
echo
read -r -p "Run rollouts to generate v3 traces now? [Y/n]: " RUN_ROLLOUTS || true
RUN_ROLLOUTS=${RUN_ROLLOUTS:-Y}
if [[ "$RUN_ROLLOUTS" =~ ^[Yy]$ || -z "$RUN_ROLLOUTS" ]]; then
  echo "Using config defaults from examples/finetuning/synth_qwen/config.toml (override below if desired)."
  # Allow quick overrides via envs
  MODEL_INPUT=$(prompt "Model id [Enter=use config]: " "")
  EPISODES_INPUT=$(prompt "Episodes [Enter=use config]: " "")
  MAX_STEPS_INPUT=$(prompt "Max steps [Enter=use config]: " "")
  DIFFICULTY_INPUT=$(prompt "Difficulty [Enter=use config]: " "")
  THINK_INPUT=$(prompt "Enable think mode? (1/0) [Enter=0]: " "0")

  if [ -n "$MODEL_INPUT" ]; then export CRAFTER_MODEL="$MODEL_INPUT"; fi
  if [ -n "$EPISODES_INPUT" ]; then export CRAFTER_EPISODES="$EPISODES_INPUT"; fi
  if [ -n "$MAX_STEPS_INPUT" ]; then export CRAFTER_MAX_STEPS="$MAX_STEPS_INPUT"; fi
  if [ -n "$DIFFICULTY_INPUT" ]; then export CRAFTER_DIFFICULTY="$DIFFICULTY_INPUT"; fi
  export CRAFTER_THINK="${THINK_INPUT:-0}"

  echo
echo "Running rollouts (v3 tracing)..."
  ensure_api_key
  uv run python -m examples.finetuning.synth_qwen.run_crafter_qwen4b
else
  echo "Skipping rollouts."
fi

# Step 2: Filter traces -> SFT JSONL
echo
read -r -p "Filter v3 traces into SFT JSONL now? [Y/n]: " RUN_FILTER || true
RUN_FILTER=${RUN_FILTER:-Y}
if [[ "$RUN_FILTER" =~ ^[Yy]$ || -z "$RUN_FILTER" ]]; then
  # Ensure DB path is correctly set for v3 traces (force set to repo-local path)
  DB_PATH_DEFAULT="$PWD/traces/v3/synth_ai.db/dbs/default/data"
  export CRAFTER_DB_URL="sqlite+aiosqlite:///$DB_PATH_DEFAULT"
  echo "Using DB: $CRAFTER_DB_URL"
  mkdir -p ft_data
  echo "You can override filter options; Enter to use config defaults."
  ACH_INPUT=$(prompt "Required achievements (space-separated) [Enter=config]: " "")
  MODELS_INPUT=$(prompt "Restrict to models (space-separated) [Enter=all]: " "")
  OUT_PATH_INPUT=$(prompt "Output JSONL path [Enter=config]: " "")
  MIN_REWARD_INPUT=$(prompt "Min total reward [Enter=config]: " "")
  MAX_COST_INPUT=$(prompt "Max total cost [Enter=config]: " "")
  MAX_TOKENS_INPUT=$(prompt "Max total tokens [Enter=config]: " "")

  if [ -n "$ACH_INPUT" ]; then export REQUIRED_ACHIEVEMENTS="$ACH_INPUT"; fi
  if [ -n "$MODELS_INPUT" ]; then export MODELS="$MODELS_INPUT"; fi
  if [ -n "$OUT_PATH_INPUT" ]; then export OUTPUT_JSONL="$OUT_PATH_INPUT"; fi
  if [ -n "$MIN_REWARD_INPUT" ]; then export MIN_TOTAL_REWARD="$MIN_REWARD_INPUT"; fi
  if [ -n "$MAX_COST_INPUT" ]; then export MAX_COST="$MAX_COST_INPUT"; fi
  if [ -n "$MAX_TOKENS_INPUT" ]; then export MAX_TOKENS="$MAX_TOKENS_INPUT"; fi

  echo
echo "Filtering traces to SFT JSONL..."
  uv run python -m examples.finetuning.synth_qwen.filter_traces_achievements
else
  echo "Skipping filter."
fi

# Step 3: Kick off SFT (learning service)
echo
read -r -p "Kick off SFT training job now? [Y/n]: " RUN_SFT || true
RUN_SFT=${RUN_SFT:-Y}
FT_MODEL_ID=""
if [[ "$RUN_SFT" =~ ^[Yy]$ || -z "$RUN_SFT" ]]; then
  echo "Enter overrides for training job; Enter to use config."
  BASE_MODEL_INPUT=$(prompt "Base model [Enter=config]: " "")
  TRAIN_JSONL_INPUT=$(prompt "Training JSONL path [Enter=config]: " "")

  if [ -n "$BASE_MODEL_INPUT" ]; then export QWEN_BASE_MODEL="$BASE_MODEL_INPUT"; fi
  if [ -n "$TRAIN_JSONL_INPUT" ]; then export QWEN_TRAINING_JSONL="$TRAIN_JSONL_INPUT"; fi

  echo
  echo "Starting SFT job..."
  ensure_api_key
  # Stream logs to terminal and save to file for parsing
  mkdir -p logs
  TS=$(date +%Y%m%d_%H%M%S)
  SFT_LOG_FILE="logs/sft_kickoff_${TS}.log"
  # Force unbuffered stdout so polling status prints live through the pipe
  PYTHONUNBUFFERED=1 uv run python -u -m examples.finetuning.synth_qwen.sft_kickoff | tee "$SFT_LOG_FILE"
  # Extract ft model id like ft:Qwen/... (no whitespace or quotes)
  if grep -qE "ft:[^[:space:]\"]+" "$SFT_LOG_FILE"; then
    FT_MODEL_ID=$(grep -Eo "ft:[^[:space:]\"]+" "$SFT_LOG_FILE" | tail -n1)
    echo "Captured fine-tuned model id: $FT_MODEL_ID"
    echo "SFT logs saved to: $SFT_LOG_FILE"
  else
    echo "Warning: could not parse fine-tuned model id from output. Logs: $SFT_LOG_FILE"
  fi
else
  echo "Skipping SFT kickoff."
fi

# Step 4: Optional rollout with fine-tuned model
echo
if [ -n "$FT_MODEL_ID" ]; then
  read -r -p "Roll out fine-tuned model '$FT_MODEL_ID' in Crafter now? [y/N]: " RUN_ROLLOUT_FT || true
  if [[ "$RUN_ROLLOUT_FT" =~ ^[Yy]$ ]]; then
    EPISODES2=$(prompt "Episodes [Enter=config]: " "")
    MAX_STEPS2=$(prompt "Max steps [Enter=config]: " "")
    DIFFICULTY2=$(prompt "Difficulty [Enter=config]: " "")
    THINK2=$(prompt "Enable think mode? (1/0) [Enter=0]: " "0")

    export CRAFTER_MODEL="$FT_MODEL_ID"
    if [ -n "$EPISODES2" ]; then export CRAFTER_EPISODES="$EPISODES2"; fi
    if [ -n "$MAX_STEPS2" ]; then export CRAFTER_MAX_STEPS="$MAX_STEPS2"; fi
    if [ -n "$DIFFICULTY2" ]; then export CRAFTER_DIFFICULTY="$DIFFICULTY2"; fi
    export CRAFTER_THINK="${THINK2:-0}"

    echo
  echo "Running rollouts with fine-tuned model..."
    uv run python -m examples.finetuning.synth_qwen.run_crafter_qwen4b
  else
    echo "Skipping rollout of fine-tuned model."
  fi
else
  echo "No fine-tuned model id available to roll out."
fi

echo
echo "Done. You can re-run this script to repeat steps as needed."
