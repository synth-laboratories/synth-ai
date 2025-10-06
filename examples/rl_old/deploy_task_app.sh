#!/usr/bin/env bash
set -euo pipefail

# Deploy SDK RL Task App (Modal) and configure secrets
# - App name: grpo-task-service-sdk (or grpo-task-service-sdk-prod when prod)
# - Secret:   crafter-environment-sdk (or crafter-environment-sdk-prod when prod)

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

# Print masked diagnostics for the ENV key being deployed (never print full value)
_env_len=${#ENVIRONMENT_API_KEY}
_env_last5="${ENVIRONMENT_API_KEY:${#ENVIRONMENT_API_KEY}-5}"
echo "ENVIRONMENT_API_KEY len=${_env_len} last5=${_env_last5}"

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

# Require an explicit secret name; environment must supply TASK_APP_SECRET_NAME
if [ -z "${TASK_APP_SECRET_NAME:-}" ]; then
  echo "TASK_APP_SECRET_NAME must be set before deploying the task app." >&2
  exit 1
fi

export TASK_APP_SECRET_NAME
SECRET_NAME="$TASK_APP_SECRET_NAME"

# Allow TASK_APP_NAME to retain existing defaults
ENV_FLAG="${SYNTH_BACKEND_URL_OVERRIDE:-${ENVIRONMENT:-${APP_ENVIRONMENT:-}}}"
ENV_FLAG=$(echo "$ENV_FLAG" | tr '[:upper:]' '[:lower:]')
if [ "$ENV_FLAG" = "prod" ] || [ "$ENV_FLAG" = "production" ]; then
  APP_NAME="${TASK_APP_NAME:-grpo-task-service-sdk-prod}"
else
  APP_NAME="${TASK_APP_NAME:-grpo-task-service-sdk}"
fi

# Ensure math task app picks up the same secret name when deployed
if [ -n "${MATH_TASK_APP_SECRET:-}" ] && [ "$MATH_TASK_APP_SECRET" != "$SECRET_NAME" ]; then
  echo "MATH_TASK_APP_SECRET must match TASK_APP_SECRET_NAME when set." >&2
  exit 1
fi
export MATH_TASK_APP_SECRET="$SECRET_NAME"

echo "Creating/updating Modal secret: $SECRET_NAME"

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
if ! uv run modal secret create "$SECRET_NAME" "${SECRET_ARGS[@]}"; then
  echo "Secret exists. Removing and recreating..."
  uv run modal secret delete "$SECRET_NAME"
  uv run modal secret create "$SECRET_NAME" "${SECRET_ARGS[@]}"
fi

echo "Deploying app: $APP_NAME"
TASK_APP_NAME="$APP_NAME" TASK_APP_SECRET_NAME="$SECRET_NAME" uv run modal deploy "$HERE/task_app.py" | tee "$HERE/.last_deploy.log"

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
