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
# Always start sqld for v3 tracing support
echo "🗄️  Starting sqld for v3 tracing support"

# First, ensure sqld is properly installed
if [ -f "./install_sqld.sh" ]; then
  ./install_sqld.sh
else
  echo "⚠️  install_sqld.sh not found, checking if sqld is already installed..."
  if ! command -v "${SQLD_BIN}" >/dev/null 2>&1; then
    echo "❌ sqld not found and install_sqld.sh is missing."
    echo "Please run: ./install_sqld.sh"
    exit 1
  fi
fi

# Check if sqld is already running
if pgrep -f "${SQLD_BIN}.*--http-listen-addr.*:${SQLD_PORT}" >/dev/null; then
  echo "✅ sqld already running on port ${SQLD_PORT}"
  echo "   Database: ${DB_FILE}"
  echo "   HTTP API: http://127.0.0.1:${SQLD_PORT}"
else
  # Check if we should use sqld for embedded replicas
  USE_SQLD_REPLICA="${USE_SQLD_REPLICA:-false}"
  TURSO_DATABASE_URL="${TURSO_DATABASE_URL:-}"
  TURSO_AUTH_TOKEN="${TURSO_AUTH_TOKEN:-}"
  
  if [ "$USE_SQLD_REPLICA" = "true" ] && [ -n "$TURSO_DATABASE_URL" ]; then
    echo "🗄️  Starting sqld with Turso replication on port ${SQLD_PORT}"
    # Use sqld with replication from Turso
    nohup "${SQLD_BIN}" \
          --db-path "embedded.db" \
          --http-listen-addr "127.0.0.1:${SQLD_PORT}" \
          --replicate-from "${TURSO_DATABASE_URL}?authToken=${TURSO_AUTH_TOKEN}" \
          > sqld.log 2>&1 &
  else
    echo "🗄️  Starting sqld (local only) on port ${SQLD_PORT}"
    # Use sqld locally for v3 tracing
    nohup "${SQLD_BIN}" \
          --db-path "${DB_FILE}" \
          --http-listen-addr "127.0.0.1:${SQLD_PORT}" \
          > sqld.log 2>&1 &
  fi
  echo "🗄️  sqld log: $(pwd)/sqld.log"
  
  # Wait a moment for sqld to start
  sleep 2
  
  # Verify sqld started successfully
  if ! pgrep -f "${SQLD_BIN}.*--http-listen-addr.*:${SQLD_PORT}" >/dev/null; then
    echo "❌ Failed to start sqld. Check sqld.log for details."
    echo ""
    echo "Common issues:"
    echo "  1. Port ${SQLD_PORT} may already be in use"
    echo "  2. sqld binary may not be executable"
    echo "  3. Database file permissions issue"
    echo ""
    echo "Last 10 lines of sqld.log:"
    tail -n 10 sqld.log 2>/dev/null || echo "  (no log file found)"
    exit 1
  else
    echo "✅ sqld started successfully!"
    echo "   Database: ${DB_FILE}"
    echo "   HTTP API: http://127.0.0.1:${SQLD_PORT}"
    echo "   Log file: $(pwd)/sqld.log"
  fi
fi

########################################
# START ENVIRONMENT SERVICE
########################################
echo ""
echo "🚀 Starting Synth-AI Environment Service on port ${ENV_PORT}"
echo ""

if [ ! -f "pyproject.toml" ] || [ ! -d "synth_ai" ]; then
  echo "❌ Must run from synth-ai project root (missing pyproject.toml or synth_ai/)"
  echo ""
  echo "Please run this script from the root of the synth-ai repository:"
  echo "  cd /path/to/synth-ai"
  echo "  ./serve.sh"
  exit 1
fi

# Check if port is already in use
if lsof -i:${ENV_PORT} >/dev/null 2>&1; then
  echo "⚠️  Port ${ENV_PORT} is already in use!"
  echo "   Another instance of the environment service may be running."
  echo "   To stop it, find the process using: lsof -i:${ENV_PORT}"
  echo ""
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export SYNTH_LOGGING="true"

echo "📦 Environment:"
echo "   Python: $(uv run python --version 2>&1)"
echo "   Working directory: $(pwd)"
echo "   PYTHONPATH: ${PYTHONPATH}"
echo ""
echo "🔄 Starting services..."
echo "   - sqld daemon: http://127.0.0.1:${SQLD_PORT}"
echo "   - Environment service: http://127.0.0.1:${ENV_PORT}"
echo ""
echo "💡 Tips:"
echo "   - Check sqld.log if database issues occur"
echo "   - Use Ctrl+C to stop all services"
echo "   - Reloading is enabled for development"
echo ""

uv run python -m uvicorn \
  synth_ai.environments.service.app:app \
  --host 0.0.0.0 \
  --port "${ENV_PORT}" \
  --log-level info \
  --reload \
  --reload-dir synth_ai
