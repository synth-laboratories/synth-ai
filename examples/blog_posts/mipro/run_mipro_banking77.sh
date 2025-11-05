#!/bin/bash
# Run MIPROv2 optimization for Banking77 against the backend

set -e

echo "🔬 Running MIPROv2 on Banking77"
echo "================================="

# Check for required environment variables
if [ -z "$SYNTH_API_KEY" ]; then
    echo "❌ Error: SYNTH_API_KEY not set"
    echo "Please get your API key from the backend and set it:"
    echo "  export SYNTH_API_KEY=your_key"
    exit 1
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "❌ Error: ENVIRONMENT_API_KEY not set"
    echo "Please set the same key used when deploying the task app:"
    echo "  export ENVIRONMENT_API_KEY=your_key"
    exit 1
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY not set"
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY=your_key"
    exit 1
fi

# Check for OpenAI API key (needed for meta-model)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "MIPROv2 uses a meta-model (gpt-4o-mini) for prompt proposals."
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY=your_key"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default to localhost backend if not specified
BACKEND_URL="${BACKEND_BASE_URL:-http://localhost:8000}"

echo "✅ SYNTH_API_KEY: ${SYNTH_API_KEY:0:20}..."
echo "✅ ENVIRONMENT_API_KEY: ${ENVIRONMENT_API_KEY:0:20}..."
echo "✅ GROQ_API_KEY: ${GROQ_API_KEY:0:20}..."
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
fi
echo "✅ Backend URL: $BACKEND_URL"
echo ""

# Navigate to repo root
cd "$(dirname "$0")/../../.."

# Check if task app is running
echo "🔍 Checking if Banking77 task app is running on http://127.0.0.1:8102..."
if ! curl -s -f -H "X-API-Key: $ENVIRONMENT_API_KEY" http://127.0.0.1:8102/health > /dev/null 2>&1; then
    echo "❌ Error: Banking77 task app is not running on http://127.0.0.1:8102"
    echo ""
    echo "Please start it first:"
    echo "  ./examples/blog_posts/mipro/deploy_banking77_task_app.sh"
    echo ""
    echo "Or in another terminal:"
    echo "  cd $(pwd)"
    echo "  uvx synth-ai deploy banking77 --runtime uvicorn --port 8102"
    exit 1
fi
echo "✅ Task app is healthy"
echo ""

# Check backend connection
echo "🔍 Checking backend connection to $BACKEND_URL..."
if ! curl -s -f "$BACKEND_URL/api/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: Cannot connect to backend at $BACKEND_URL"
    echo "Make sure the backend is running."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Backend is healthy"
fi
echo ""

echo "🚀 Starting MIPROv2 training..."
echo "Config: examples/blog_posts/mipro/configs/banking77_mipro_local.toml"
echo ""
echo "MIPROv2 Flow:"
echo "  1. Bootstrap Phase: Evaluate baseline on seeds [0-4], collect few-shot examples"
echo "  2. Optimization Loop: 16 iterations × 6 variants = 96 evaluations"
echo "  3. Final Evaluation: Test on held-out seeds [10-19]"
echo ""

# Run the training
uvx synth-ai train \
    --type prompt_learning \
    --config examples/blog_posts/mipro/configs/banking77_mipro_local.toml \
    --backend "$BACKEND_URL" \
    --poll

echo ""
echo "✅ MIPROv2 training complete!"

