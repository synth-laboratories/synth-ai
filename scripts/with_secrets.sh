#!/usr/bin/env bash
#
# Run a command with secrets injected from Infisical (preferred) with a .env fallback.
#
# Usage:
#   ./scripts/with_secrets.sh -- <command...>
#
# Controls:
#   INFISICAL_DISABLED=1   Force .env fallback (no Infisical).
#   ENV_FILE=.env          Which env file to source in fallback mode.
#
set -euo pipefail

if [[ "${1:-}" != "--" ]]; then
  echo "Usage: $0 -- <command...>" >&2
  exit 2
fi
shift

if [[ "$#" -eq 0 ]]; then
  echo "Error: missing command" >&2
  exit 2
fi

env_file="${ENV_FILE:-.env}"

run_with_env_file() {
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  else
    echo "Warning: $env_file not found; running without injected secrets." >&2
  fi
  exec "$@"
}

if [[ "${INFISICAL_DISABLED:-}" == "1" || "${INFISICAL_DISABLED:-}" == "true" ]]; then
  run_with_env_file "$@"
fi

if ! command -v infisical >/dev/null 2>&1; then
  echo "Warning: infisical CLI not found; falling back to $env_file." >&2
  run_with_env_file "$@"
fi

if infisical run --help 2>/dev/null | grep -q -- "--command" 2>/dev/null; then
  joined=""
  for arg in "$@"; do
    joined+=$(printf ' %q' "$arg")
  done
  exec infisical run --command "${joined# }"
fi

exec infisical run -- "$@"

