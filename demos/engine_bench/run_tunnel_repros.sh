#!/usr/bin/env bash
#
# Run all tunnel issue reproduction scripts.
#
# Usage:
#   cd /Users/joshpurtell/Documents/GitHub/synth-ai
#   ./demos/engine_bench/run_tunnel_repros.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"

echo "============================================================"
echo "TUNNEL ISSUES REPRODUCTION SUITE"
echo "============================================================"
echo "Repo: $REPO_ROOT"
echo "Python: $PYTHON"
echo ""

# Track results
declare -a RESULTS
PASSED=0
FAILED=0

run_repro() {
    local name="$1"
    local script="$2"
    local expected_exit="$3"  # 0=expect pass, 1=expect fail (reproduced)
    
    echo ""
    echo "============================================================"
    echo "Running: $name"
    echo "Script: $script"
    echo "============================================================"
    
    set +e
    "$PYTHON" "$script"
    local exit_code=$?
    set -e
    
    if [ "$expected_exit" = "0" ]; then
        # Expecting pass
        if [ $exit_code -eq 0 ]; then
            echo "→ EXPECTED PASS: ✅"
            RESULTS+=("✅ $name (passed as expected)")
            ((PASSED++))
        else
            echo "→ UNEXPECTED FAIL: ❌"
            RESULTS+=("❌ $name (unexpected failure)")
            ((FAILED++))
        fi
    else
        # Expecting fail (issue reproduced)
        if [ $exit_code -ne 0 ]; then
            echo "→ ISSUE REPRODUCED: ✅ (expected)"
            RESULTS+=("✅ $name (issue reproduced)")
            ((PASSED++))
        else
            echo "→ ISSUE NOT REPRODUCED: ❓"
            RESULTS+=("❓ $name (issue NOT reproduced - maybe fixed?)")
            ((FAILED++))
        fi
    fi
}

# Run control test (expected to PASS)
run_repro "Control: cloudflared CLI" \
    "demos/engine_bench/repro_control_cloudflared_cli.py" \
    "0"

# Run issue repros (expected to FAIL = issue reproduced)
run_repro "Issue #1: CloudflareManagedLease" \
    "demos/engine_bench/repro_issue1_lease_not_found.py" \
    "1"

run_repro "Issue #2: CloudflareManagedTunnel" \
    "demos/engine_bench/repro_issue2_rotate_failed.py" \
    "1"

run_repro "Issue #3: CloudflareQuickTunnel" \
    "demos/engine_bench/repro_issue3_quick_timeout.py" \
    "1"

# Summary
echo ""
echo "============================================================"
echo "REPRODUCTION RESULTS SUMMARY"
echo "============================================================"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo "Total: $PASSED expected, $FAILED unexpected"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Some results were unexpected. Check the output above."
    exit 1
else
    echo ""
    echo "All results match expectations."
    echo ""
    echo "Note: Issue #4 (idle timeout) is blocked until issues #1-3 are fixed."
    echo "Once a tunnel backend works, run:"
    echo "  .venv/bin/python demos/engine_bench/repro_issue4_idle_timeout.py"
fi
