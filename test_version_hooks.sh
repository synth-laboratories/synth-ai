#!/bin/bash
# Test script for version hooks

set -e

echo "🧪 Testing Git Version Hooks"
echo "=============================="
echo ""

# Save current state
CURRENT_BRANCH=$(git branch --show-current)
echo "📌 Current branch: $CURRENT_BRANCH"
echo ""

# Test 1: Pre-commit on main (should pass)
echo "Test 1: Pre-commit hook on main branch"
echo "---------------------------------------"
.git/hooks/pre-commit
if [ $? -eq 0 ]; then
    echo "✅ PASSED: Pre-commit on main branch"
else
    echo "❌ FAILED: Pre-commit on main branch"
fi
echo ""

# Test 2: Simulate nightly branch check
echo "Test 2: Checking version comparison logic"
echo "-------------------------------------------"

# Get versions
MAIN_VERSION=$(git show origin/main:pyproject.toml | grep '^version' | sed -E 's/version = "(.*)"/\1/')
NIGHTLY_VERSION=$(git show origin/nightly:pyproject.toml | grep '^version' | sed -E 's/version = "(.*)"/\1/')
PYPI_VERSION=$(curl -s https://pypi.org/pypi/synth-ai/json | python3 -c "import sys, json; print(json.load(sys.stdin)['info']['version'])")

echo "   Main version:    $MAIN_VERSION"
echo "   Nightly version: $NIGHTLY_VERSION"
echo "   PyPI version:    $PYPI_VERSION"
echo ""

# Compare versions using Python
python3 << EOF
def parse_version(v):
    return tuple(map(int, v.split('.')))

main_v = parse_version('$MAIN_VERSION')
nightly_v = parse_version('$NIGHTLY_VERSION')
pypi_v = parse_version('$PYPI_VERSION')

print("Version comparisons:")
if nightly_v >= main_v:
    print(f"   ✅ Nightly ($NIGHTLY_VERSION) >= Main ($MAIN_VERSION)")
else:
    print(f"   ❌ Nightly ($NIGHTLY_VERSION) < Main ($MAIN_VERSION)")

if nightly_v >= pypi_v:
    print(f"   ✅ Nightly ($NIGHTLY_VERSION) >= PyPI ($PYPI_VERSION)")
else:
    print(f"   ⚠️  Nightly ($NIGHTLY_VERSION) < PyPI ($PYPI_VERSION)")
EOF

echo ""
echo "Test 3: Hook behavior on different branches"
echo "---------------------------------------------"

# If we're already on nightly, test directly
if [ "$CURRENT_BRANCH" = "nightly" ]; then
    echo "   Testing on nightly branch (current)..."
    .git/hooks/pre-commit
    if [ $? -eq 0 ]; then
        echo "   ✅ Pre-commit passed on nightly"
    else
        echo "   ❌ Pre-commit failed on nightly"
    fi
else
    echo "   ℹ️  Not on nightly branch - skipping live test"
    echo "   To test nightly checks:"
    echo "      git checkout nightly"
    echo "      .git/hooks/pre-commit"
fi

echo ""
echo "=============================="
echo "✅ Hook testing complete!"
echo ""
echo "📝 Hook capabilities:"
echo "   • Validates semantic versioning format"
echo "   • Prevents nightly from being behind main"
echo "   • Warns if nightly is behind PyPI"
echo "   • Runs on both commit (pre-commit) and push (pre-push)"
echo ""
