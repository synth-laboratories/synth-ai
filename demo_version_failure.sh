#!/bin/bash
# Demonstration of hook failure scenarios

echo "🎭 Demonstrating Version Hook Failure Scenarios"
echo "================================================"
echo ""

echo "Scenario 1: Nightly version behind main"
echo "----------------------------------------"
cat << 'EOF'

If you're on nightly branch with version 0.4.5, but main has 0.4.6:

🔍 Running version consistency checks for nightly branch...
   Current version: 0.4.5
   Main version:    0.4.6

❌ ERROR: Nightly version (0.4.5) is behind main version (0.4.6)
   Nightly must always be >= main version

   To fix:
   1. Update version in pyproject.toml to >= 0.4.6
   2. Try committing again

The commit would be REJECTED.
EOF

echo ""
echo ""
echo "Scenario 2: Invalid version format"
echo "-----------------------------------"
cat << 'EOF'

If version in pyproject.toml is set to "invalid-version":

❌ ERROR: Invalid version format: invalid-version
Version must follow semantic versioning: X.Y.Z

The commit would be REJECTED.
EOF

echo ""
echo ""
echo "Scenario 3: Nightly same as main (warning only)"
echo "------------------------------------------------"
cat << 'EOF'

If you're on nightly branch with version 0.4.6, same as main:

🔍 Running version consistency checks for nightly branch...
   Current version: 0.4.6
   Main version:    0.4.6
   ⚠️  WARNING: Nightly and main are at the same version
   PyPI version:    0.4.6
   ⚠️  WARNING: Nightly matches PyPI version

✅ Version checks passed!

The commit would be ALLOWED but with warnings.
EOF

echo ""
echo ""
echo "================================================"
echo "✅ Demo complete!"
echo ""
echo "📚 Summary:"
echo "   • Hooks prevent nightly from falling behind main"
echo "   • Hooks validate semantic versioning format"
echo "   • Hooks warn when nightly matches or lags PyPI"
echo "   • Hooks run on commit and push for safety"
echo ""
