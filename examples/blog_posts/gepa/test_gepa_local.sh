#!/bin/bash
# Quick test script for GEPA Banking77 prompt learning
# Tests against local backend on port 8000

set -e

echo "🚀 Testing GEPA Prompt Learning for Banking77"
echo "=============================================="

# Check required environment variables
if [ -z "$SYNTH_API_KEY" ]; then
    echo "❌ ERROR: SYNTH_API_KEY not set"
    exit 1
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "❌ ERROR: ENVIRONMENT_API_KEY not set"
    exit 1
fi

# Set backend URL (default to localhost:8000)
BACKEND_URL="${BACKEND_BASE_URL:-http://localhost:8000}"
echo "📍 Backend URL: $BACKEND_URL"

# Check backend is accessible
echo "🔍 Checking backend health..."
if curl -s -f "$BACKEND_URL/api/health" > /dev/null 2>&1; then
    echo "✅ Backend is accessible"
else
    echo "❌ ERROR: Backend not accessible at $BACKEND_URL"
    echo "   Make sure backend is running on port 8000"
    exit 1
fi

# Check task app is accessible
TASK_APP_URL="${TASK_APP_URL:-http://127.0.0.1:8102}"
echo "🔍 Checking task app health..."
if curl -s -f -H "X-API-Key: $ENVIRONMENT_API_KEY" "$TASK_APP_URL/health" > /dev/null 2>&1; then
    echo "✅ Task app is accessible"
else
    echo "⚠️  WARNING: Task app not accessible at $TASK_APP_URL"
    echo "   You may need to deploy it first:"
    echo "   uvx synth-ai deploy banking77 --runtime uvicorn --port 8102"
fi

# Run GEPA training
echo ""
echo "🎯 Starting GEPA prompt optimization..."
echo ""

CONFIG_FILE="examples/blog_posts/gepa/configs/banking77_gepa_local.toml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

uvx synth-ai train \
  --type prompt_learning \
  --config "$CONFIG_FILE" \
  --backend "$BACKEND_URL" \
  --poll \
  --poll-timeout 3600

echo ""
echo "✅ GEPA training completed!"

