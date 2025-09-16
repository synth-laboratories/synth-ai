#!/usr/bin/env bash
set -euo pipefail

# Ensure Modal secret 'crafter-environment-sdk' exists with ENVIRONMENT_API_KEY.
# - If ENVIRONMENT_API_KEY missing in examples/rl/.env, mint one and persist.
# - Sources examples/rl/.env into the current process when done (prints export hints).

HERE=$(cd "$(dirname "$0")" && pwd)
ENV_FILE="$HERE/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Creating $ENV_FILE"
  touch "$ENV_FILE"
fi

# Load existing values (if any)
set +u
if [ -s "$ENV_FILE" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "$ENV_FILE" | xargs -I{} echo {})
fi
set -u

# Ensure ENVIRONMENT_API_KEY
if [ -z "${ENVIRONMENT_API_KEY:-}" ]; then
  if command -v openssl >/dev/null 2>&1; then
    ENVIRONMENT_API_KEY=$(openssl rand -hex 32)
  else
    ENVIRONMENT_API_KEY=$(dd if=/dev/urandom bs=32 count=1 2>/dev/null | base64 | tr -d '=+/\n' | cut -c1-64)
  fi
  echo "Minted new ENVIRONMENT_API_KEY"
  if grep -q '^ENVIRONMENT_API_KEY=' "$ENV_FILE"; then
    # Replace existing line
    sed -i.bak "s/^ENVIRONMENT_API_KEY=.*/ENVIRONMENT_API_KEY=$ENVIRONMENT_API_KEY/" "$ENV_FILE" || true
  else
    echo "ENVIRONMENT_API_KEY=$ENVIRONMENT_API_KEY" >> "$ENV_FILE"
  fi
fi

# Optional passthroughs
OPENAI_ARG=()
if [ -n "${OPENAI_API_KEY:-}" ]; then
  OPENAI_ARG=(OPENAI_API_KEY="$OPENAI_API_KEY")
fi
SYNTH_ARG=()
if [ -n "${SYNTH_API_KEY:-}" ]; then
  SYNTH_ARG=(SYNTH_API_KEY="$SYNTH_API_KEY")
fi

echo "Creating/updating Modal secret: crafter-environment-sdk"
modal secret create crafter-environment-sdk \
  ENVIRONMENT_API_KEY="$ENVIRONMENT_API_KEY" \
  ${OPENAI_ARG:+${OPENAI_ARG[@]}} \
  ${SYNTH_ARG:+${SYNTH_ARG[@]}} || true

echo "Done. Sourced values available in $ENV_FILE."
echo "To load them now in your shell:"
echo "  set -a; source $ENV_FILE; set +a"

