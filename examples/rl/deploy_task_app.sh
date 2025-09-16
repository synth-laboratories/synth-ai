#!/usr/bin/env bash
set -euo pipefail

# Deploy SDK RL Task App (Modal) and configure secrets
# - App name: grpo-task-service-sdk
# - Secret:   crafter-environment-sdk

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

ENV_FILE="$HERE/.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "$ENV_FILE" | xargs -I{} echo {})
fi

if [ -z "${ENVIRONMENT_API_KEY:-}" ]; then
  echo "ENVIRONMENT_API_KEY is required (set in examples/rl/.env)" >&2
  exit 1
fi

# Use `uv run modal` directly; no global CLI install required
export PATH="$HOME/.local/bin:$PATH"

# OPENAI_API_KEY optional unless using proxy
OPENAI_ARG=()
if [ -n "${OPENAI_API_KEY:-}" ]; then
  OPENAI_ARG=(OPENAI_API_KEY="$OPENAI_API_KEY")
fi

# SYNTH_API_KEY optional
SYNTH_ARG=()
if [ -n "${SYNTH_API_KEY:-}" ]; then
  SYNTH_ARG=(SYNTH_API_KEY="$SYNTH_API_KEY")
fi

echo "Creating/updating Modal secret: crafter-environment-sdk"

# Build secret args dynamically to avoid sending empty vars
SECRET_ARGS=("ENVIRONMENT_API_KEY=$ENVIRONMENT_API_KEY")
if [ -n "${OPENAI_API_KEY:-}" ]; then
  SECRET_ARGS+=("OPENAI_API_KEY=$OPENAI_API_KEY")
else
  echo "Warning: OPENAI_API_KEY not set; tool-based models will not work until provided" >&2
fi
if [ -n "${SYNTH_API_KEY:-}" ]; then
  SECRET_ARGS+=("SYNTH_API_KEY=$SYNTH_API_KEY")
fi

# Try create; if it exists, update via `secret create` (set command doesn't exist)
if ! uv run modal secret create crafter-environment-sdk "${SECRET_ARGS[@]}"; then
  echo "Secret exists. Removing and recreating..."
  uv run modal secret delete crafter-environment-sdk
  uv run modal secret create crafter-environment-sdk "${SECRET_ARGS[@]}"
fi

echo "Deploying app: grpo-task-service-sdk"
uv run modal deploy "$HERE/task_app.py" | tee "$HERE/.last_deploy.log"

# Extract deployed Task App URL and persist to .env for downstream scripts
TASK_URL=$(grep -Eo 'https://[^ ]+\.modal\.run' "$HERE/.last_deploy.log" | tail -1 || true)
if [ -n "$TASK_URL" ]; then
  if grep -q '^TASK_APP_BASE_URL=' "$ENV_FILE"; then
    sed -i.bak "s#^TASK_APP_BASE_URL=.*#TASK_APP_BASE_URL=$TASK_URL#" "$ENV_FILE" || true
  else
    echo "TASK_APP_BASE_URL=$TASK_URL" >> "$ENV_FILE"
  fi
  echo "Saved TASK_APP_BASE_URL to $ENV_FILE"
else
  echo "Warning: could not auto-detect Task App URL from deploy output. Run 'modal app list' to get it." >&2
fi

# Persist resolved backend URL for convenience
BACKEND_URL=$(uv run python -c 'from examples.common.backend import resolve_backend_url;print(resolve_backend_url())')
if [ -n "$BACKEND_URL" ]; then
  if grep -q '^PROD_BACKEND_URL=' "$ENV_FILE"; then
    sed -i.bak "s#^PROD_BACKEND_URL=.*#PROD_BACKEND_URL=$BACKEND_URL#" "$ENV_FILE" || true
  else
    echo "PROD_BACKEND_URL=$BACKEND_URL" >> "$ENV_FILE"
  fi
  echo "Saved PROD_BACKEND_URL to $ENV_FILE"
fi

echo "Done. You can now load env with: set -a; source $ENV_FILE; set +a"
