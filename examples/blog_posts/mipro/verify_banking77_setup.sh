#!/bin/bash
# Verify Banking77 MIPROv2 setup

set -e

echo "🔍 Verifying Banking77 MIPROv2 Setup"
echo "====================================="
echo ""

# Check environment variables
echo "📋 Environment Variables:"
MISSING_VARS=0

if [ -z "$SYNTH_API_KEY" ]; then
    echo "  ❌ SYNTH_API_KEY: not set"
    MISSING_VARS=$((MISSING_VARS + 1))
else
    echo "  ✅ SYNTH_API_KEY: ${SYNTH_API_KEY:0:20}..."
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "  ❌ GROQ_API_KEY: not set"
    MISSING_VARS=$((MISSING_VARS + 1))
else
    echo "  ✅ GROQ_API_KEY: ${GROQ_API_KEY:0:20}..."
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "  ⚠️  OPENAI_API_KEY: not set (required for meta-model)"
    MISSING_VARS=$((MISSING_VARS + 1))
else
    echo "  ✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "  ❌ ENVIRONMENT_API_KEY: not set"
    MISSING_VARS=$((MISSING_VARS + 1))
else
    echo "  ✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
fi

echo ""

# Check backend
echo "🌐 Backend Connection:"
BACKEND_URL="${BACKEND_BASE_URL:-http://localhost:8000}"
if curl -s -f "$BACKEND_URL/api/health" > /dev/null 2>&1; then
    echo "  ✅ Backend is healthy at $BACKEND_URL"
else
    echo "  ❌ Cannot connect to backend at $BACKEND_URL"
    MISSING_VARS=$((MISSING_VARS + 1))
fi
echo ""

# Check task app
echo "📱 Task App Connection:"
if curl -s -f -H "X-API-Key: ${ENVIRONMENT_API_KEY:-dummy}" http://127.0.0.1:8102/health > /dev/null 2>&1; then
    echo "  ✅ Banking77 task app is running on http://127.0.0.1:8102"
else
    echo "  ❌ Banking77 task app is not running on http://127.0.0.1:8102"
    echo "     Start it with: ./examples/blog_posts/mipro/deploy_banking77_task_app.sh"
    MISSING_VARS=$((MISSING_VARS + 1))
fi
echo ""

# Check config files
echo "📄 Configuration Files:"
CONFIG_DIR="examples/blog_posts/mipro/configs"
if [ -f "$CONFIG_DIR/banking77_mipro_local.toml" ]; then
    echo "  ✅ banking77_mipro_local.toml exists"
else
    echo "  ❌ banking77_mipro_local.toml not found"
    MISSING_VARS=$((MISSING_VARS + 1))
fi

if [ -f "$CONFIG_DIR/banking77_mipro_test.toml" ]; then
    echo "  ✅ banking77_mipro_test.toml exists"
else
    echo "  ❌ banking77_mipro_test.toml not found"
    MISSING_VARS=$((MISSING_VARS + 1))
fi
echo ""

# Summary
if [ $MISSING_VARS -eq 0 ]; then
    echo "✅ All checks passed! Ready to run MIPROv2 optimization."
    echo ""
    echo "Next step:"
    echo "  ./examples/blog_posts/mipro/run_mipro_banking77.sh"
    exit 0
else
    echo "❌ Setup incomplete. Please fix the issues above."
    exit 1
fi

