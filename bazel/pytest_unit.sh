#!/usr/bin/env bash
set -euo pipefail

HOME_DIR="${HOME:-/home/runner}"
export PATH="${HOME_DIR}/.cargo/bin:${HOME_DIR}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
if ! command -v uv >/dev/null 2>&1; then
  for candidate in /opt/hostedtoolcache/uv/*/*/uv "${HOME_DIR}/.local/bin/uv" /home/runner/.local/bin/uv; do
    if [[ -x "${candidate}" ]]; then
      export PATH="$(dirname "${candidate}"):${PATH:-}"
      break
    fi
  done
fi

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

uv run pytest tests/unit -v
