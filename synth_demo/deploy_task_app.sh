#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
APP="$HERE/task_app.py"
if [ -f "$HERE/.env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "$HERE/.env" | xargs -I{} echo {})
fi
uv run modal deploy "$APP" | tee "$HERE/.last_deploy.log"
URL=$(grep -Eo 'https://[^ ]+\.modal\.run' "$HERE/.last_deploy.log" | tail -1 || true)
if [ -n "$URL" ]; then
  if grep -q '^TASK_APP_BASE_URL=' "$HERE/.env" 2>/dev/null; then
    sed -i.bak "s#^TASK_APP_BASE_URL=.*#TASK_APP_BASE_URL=$URL#" "$HERE/.env" || true
  else
    echo "TASK_APP_BASE_URL=$URL" >> "$HERE/.env"
  fi
  echo "Saved TASK_APP_BASE_URL to $HERE/.env"
fi
