#!/bin/bash
# End-to-end Modal tracing verification
# This will trigger a request and extract the key log lines

set -e

echo "========================================================================"
echo "MODAL TRACING END-TO-END VERIFICATION"
echo "========================================================================"
echo ""
echo "Step 1: Triggering test rollout request..."
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run test_modal_tracing_final.py > /dev/null 2>&1 || true

echo "Step 2: Waiting for logs to flush (3 seconds)..."
sleep 3

echo ""
echo "========================================================================"
echo "VERIFICATION CHECKLIST"
echo "========================================================================"
echo ""
echo "Please check your Modal serve terminal for these logs:"
echo ""
echo "✅ 1. [TRACING_V3_CONFIG_LOADED] Python=3.11 MODAL_IS_REMOTE=1"
echo "      ^-- Confirms new code is running and Modal env detected"
echo ""
echo "✅ 2. [TRACE_CONFIG] Modal detection: True (MODAL_IS_REMOTE=1)"
echo "      ^-- Confirms Modal detection logic works"
echo ""
echo "✅ 3. [TRACE_CONFIG] Using Modal SQLite: file:/tmp/synth_traces.db"
echo "      ^-- Confirms SQLite fallback is chosen"
echo ""
echo "✅ 4. No 'RuntimeError: Tracing backend not reachable'"
echo "      ^-- Confirms health check was skipped (SQLite doesn't need sqld)"
echo ""
echo "✅ 5. Rollout completes without tracing errors"
echo "      ^-- Confirms end-to-end flow works"
echo ""
echo "========================================================================"
echo "If you see ALL 5 checkmarks in your Modal logs, the fix is VERIFIED! ✨"
echo "========================================================================"


