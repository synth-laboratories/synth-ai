#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
if [[ "${SCRIPT_PATH}" == *".runfiles/"* ]]; then
  RUNFILES_ROOT="${SCRIPT_PATH%%.runfiles/*}.runfiles"
  REL_PATH="${SCRIPT_PATH#${RUNFILES_ROOT}/}"
  REPO_NAME="${REL_PATH%%/*}"
  ROOT="${RUNFILES_ROOT}/${REPO_NAME}"
else
  ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
fi
cd "${ROOT}"

uv run pytest -m "not slow and not integration and not private" -v
