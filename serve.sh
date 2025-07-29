#!/bin/bash
# Synth-AI Environment Service Startup Script
# Starts Turso sqld (SQLite server) and the Crafter environment service.

set -e

########################################
# CONFIG
########################################
DB_FILE="synth_ai.db"
SQLD_PORT="8080"
ENV_PORT="8901"
SQLD_BIN="sqld"

########################################
# START TURSO SQLD (background)
########################################
echo "🗄️  Checking Turso sqld…"

if ! command -v "${SQLD_BIN}" >/dev/null 2>&1; then
  echo "❌ '${SQLD_BIN}' not found on PATH."

  echo -n "💡 Do you want to install sqld now? (y/n): "
  read -r RESP
  if [[ "$RESP" == "y" || "$RESP" == "Y" ]]; then
    echo "📦 Installing sqld from GitHub release (latest Linux/macOS)…"
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then ARCH="x86_64"; fi
    if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then ARCH="aarch64"; fi

    DL_URL="https://github.com/tursodatabase/libsql/releases/latest/download/sqld-${OS}-${ARCH}"
    INSTALL_PATH="/usr/local/bin/sqld"

    echo "⬇️  Downloading from: $DL_URL"
    curl -L "$DL_URL" -o /tmp/sqld
    chmod +x /tmp/sqld

    if [ -w "/usr/local/bin" ]; then
      mv /tmp/sqld "$INSTALL_PATH"
      echo "✅ Installed sqld to $INSTALL_PATH"
    else
      echo "🔐 sudo required to move sqld to $INSTALL_PATH"
      sudo mv /tmp/sqld "$INSTALL_PATH"
      echo "✅ Installed sqld to $INSTALL_PATH (via sudo)"
    fi
  else
    echo "❌ Cannot proceed without 'sqld'. Exiting."
    exit 1
  fi
fi

if pgrep -f "${SQLD_BIN}.*--http-listen-addr.*:${SQLD_PORT}" >/dev/null; then
  echo "🗄️  sqld already running on port ${SQLD_PORT}"
else
  echo "🗄️  Starting sqld on port ${SQLD_PORT} (DB: ${DB_FILE})"
  nohup "${SQLD_BIN}" \
        --database "${DB_FILE}" \
        --http-listen-addr "127.0.0.1:${SQLD_PORT}" \
        --pg-listen-addr "127.0.0.1:$((${SQLD_PORT}+1))" \
        > sqld.log 2>&1 &
  echo "🗄️  sqld log: $(pwd)/sqld.log"
fi

########################################
# START ENVIRONMENT SERVICE
########################################
echo ""
echo "🚀 Starting Synth-AI Environment Service on port ${ENV_PORT}"
echo ""

if [ ! -f "pyproject.toml" ] || [ ! -d "synth_ai" ]; then
  echo "❌ Must run from synth-ai project root (missing pyproject.toml or synth_ai/)"
  exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export SYNTH_LOGGING="true"

uv run python -m uvicorn \
  synth_ai.environments.service.app:app \
  --host 0.0.0.0 \
  --port "${ENV_PORT}" \
  --log-level info \
  --reload \
  --reload-dir synth_ai
