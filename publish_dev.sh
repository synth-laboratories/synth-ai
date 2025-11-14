#!/bin/bash
set -e

# Build synth-ai
echo "Building synth-ai..."
cd /Users/joshpurtell/Documents/GitHub/synth-ai
python3 -m build

# Check if build succeeded
if [ ! -f dist/synth_ai-0.2.23.dev4-*.whl ]; then
    echo "❌ Build failed - wheel not found"
    exit 1
fi

echo "✓ Build successful"

# Load credentials from monorepo/.env.dev
echo "Loading PyPI credentials..."
cd /Users/joshpurtell/Documents/GitHub/monorepo
if [ -f .env.dev ]; then
    set -a
    source .env.dev
    set +a
    echo "✓ Credentials loaded"
else
    echo "❌ .env.dev not found"
    exit 1
fi

# Publish to PyPI
echo "Publishing to PyPI..."
cd /Users/joshpurtell/Documents/GitHub/synth-ai
python3 -m twine upload dist/synth_ai-0.2.23.dev4-*.whl --repository pypi

echo "✓ Published successfully"

# Install in backend
echo "Installing in backend..."
cd /Users/joshpurtell/Documents/GitHub/monorepo/backend
if command -v uv &> /dev/null; then
    uv pip install --upgrade "synth-ai>=0.2.23.dev4"
else
    pip install --upgrade "synth-ai>=0.2.23.dev4"
fi

echo "✓ Installation complete"

# Verify import
echo "Verifying import..."
python3 -c "import synth_ai.cloudflare; print('✓ synth_ai.cloudflare imported successfully'); print(f'open_quick_tunnel available: {hasattr(synth_ai.cloudflare, \"open_quick_tunnel\")}')"

echo ""
echo "✅ All done! synth-ai 0.2.23.dev4 is published and installed."

