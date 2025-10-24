#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_file="${script_dir}/.last_deploy.log"
rm -f "$log_file"

: "${ENVIRONMENT_API_KEY:=}"

repo_root="$(cd "${script_dir}/../../../.." && pwd)"
fallback_script="${repo_root}/examples/rl/deploy_task_app.sh"

if [[ -f "${fallback_script}" ]]; then
  echo "Using ${fallback_script} via 'uv run'" | tee -a "$log_file"
  (cd "${repo_root}/examples/rl" && ENVIRONMENT_API_KEY="${ENVIRONMENT_API_KEY}" uv run bash "${fallback_script}" | tee -a "$log_file")
else
  echo "ERROR: Deploy script not found at ${fallback_script}. Pass --script /path/to/deploy_task_app.sh" | tee -a "$log_file"
  exit 1
fi

echo "Deploy finished. Inspect $log_file for the public URL (…modal.run)." | tee -a "$log_file"

