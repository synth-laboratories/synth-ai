#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing .env in $ROOT_DIR (needs ENVIRONMENT_API_KEY, GROQ_API_KEY)" >&2
  exit 1
fi

echo "Loading .env and preparing environment..."
set -a; source .env; set +a

# Ensure GROQ key is present; avoid accidentally using OPENAI key for Groq proxy
if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY not set in .env — Groq proxy calls will fail." >&2
  exit 2
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "Note: OPENAI_API_KEY is set; will unset it for the server process to avoid wrong Authorization header." >&2
fi

# Force local serve base URL unless explicitly disabled
FORCE_LOCAL=${FORCE_LOCAL:-1}
if [[ "${FORCE_LOCAL}" != "0" ]]; then
  export TASK_APP_URL=http://localhost:8001
fi

# Ensure nothing else is bound to the target port; terminate stale servers proactively
PORT=${PORT:-8001}
EXISTING_PIDS=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${EXISTING_PIDS}" ]]; then
  echo "Port ${PORT} in use by: ${EXISTING_PIDS}. Terminating stale processes..."
  # Try graceful termination first, then force if they linger
  kill ${EXISTING_PIDS} 2>/dev/null || true
  sleep 1
  STILL_RUNNING=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
  if [[ -n "${STILL_RUNNING}" ]]; then
    echo "Processes still holding port ${PORT}: ${STILL_RUNNING}. Sending SIGKILL..."
    kill -9 ${STILL_RUNNING} 2>/dev/null || true
    sleep 1
  fi
fi

echo "Starting grpo-crafter on ${TASK_APP_URL} ..."
# Non-interactive answers for tracing prompts; unset OPENAI_API_KEY in child env
nohup env -u OPENAI_API_KEY bash -lc 'printf "Y\n\n\n" | uvx synth-ai serve grpo-crafter --port 8001 --env-file .env' > /tmp/crafter_serve.log 2>&1 &
PID=$!
echo "PID: $PID | Logs: /tmp/crafter_serve.log"
sleep 1
# Wait up to ~20s for health
for i in {1..20}; do
  HEALTH_JSON=$(curl -s -H "X-API-Key: ${ENVIRONMENT_API_KEY:-}" "${TASK_APP_URL}/health") || true
  if [[ "$HEALTH_JSON" == *"healthy"* ]]; then
    echo "Health: $HEALTH_JSON"
    break
  fi
  sleep 1
done

# Sanity-check Groq proxy with POST (avoid 405); expect 200/400/401, not 404/405
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${GROQ_API_KEY}" \
  -H "X-API-Key: ${ENVIRONMENT_API_KEY:-}" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"ping"}]}' \
  "${TASK_APP_URL}/proxy/groq/v1/chat/completions")
echo "Groq proxy POST /proxy/groq/v1/chat/completions -> HTTP ${HTTP_CODE} (expected 200/400/401, not 404/405)"
if [[ "${HTTP_CODE}" == "404" || "${HTTP_CODE}" == "405" ]]; then
  echo "Proxy route not found or wrong method. Is the task app exposing the Groq proxy?" >&2
  tail -n +1 /tmp/crafter_serve.log
  exit 3
fi
tail -f /tmp/crafter_serve.log
