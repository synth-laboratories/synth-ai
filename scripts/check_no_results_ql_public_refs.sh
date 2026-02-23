#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATTERN='results[[:space:]_-]*ql'

# Guard public-facing surfaces only.
SCAN_PATHS=(
  "$ROOT/README.md"
  "$ROOT/contracts"
  "$ROOT/examples"
  "$ROOT/synth_ai"
)

if rg --line-number --ignore-case --fixed-strings "Results QL" "${SCAN_PATHS[@]}"; then
  echo
  echo "Error: internal 'Results QL' wording found in public-facing surfaces."
  echo "Please remove those references before merging."
  exit 1
fi

if rg --line-number --ignore-case --regexp "$PATTERN" "${SCAN_PATHS[@]}"; then
  echo
  echo "Error: results-ql style phrasing found in public-facing surfaces."
  echo "Please keep query-layer internals private in SDK/docs/examples."
  exit 1
fi

echo "OK: no public Results QL references found."
