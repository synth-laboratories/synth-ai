#!/bin/bash
# Verify Banking77 setup is working

set -e

echo "🔍 Verifying Banking77 Setup"
echo "============================="
echo ""

cd "$(dirname "$0")/../../.."

echo "1️⃣ Checking Python import..."
python3 -c "
try:
    from examples.task_apps.banking77.banking77_task_app import build_config
    print('   ✅ Task app imports successfully')
    config = build_config()
    print(f'   ✅ Config built: app_id={config.app_id}')
    print(f'   ✅ Task name: {config.name}')
except ImportError as e:
    print(f'   ❌ Import error: {e}')
    print('   💡 Run: uv pip install -e .')
    exit(1)
except Exception as e:
    print(f'   ❌ Error: {e}')
    exit(1)
"

echo ""
echo "2️⃣ Checking CLI registration..."
if uvx synth-ai task-app list 2>/dev/null | grep -q "banking77"; then
    echo "   ✅ Banking77 registered with CLI"
else
    echo "   ⚠️  Banking77 not found in task-app list"
    echo "   💡 This is OK if you haven't run 'uv pip install -e .' yet"
fi

echo ""
echo "3️⃣ Checking helper scripts..."
if [ -x "./examples/blog_posts/gepa/deploy_banking77_task_app.sh" ]; then
    echo "   ✅ deploy_banking77_task_app.sh is executable"
else
    echo "   ❌ deploy_banking77_task_app.sh is not executable"
    echo "   💡 Run: chmod +x ./examples/blog_posts/gepa/deploy_banking77_task_app.sh"
fi

if [ -x "./examples/blog_posts/gepa/run_gepa_banking77.sh" ]; then
    echo "   ✅ run_gepa_banking77.sh is executable"
else
    echo "   ❌ run_gepa_banking77.sh is not executable"
    echo "   💡 Run: chmod +x ./examples/blog_posts/gepa/run_gepa_banking77.sh"
fi

echo ""
echo "4️⃣ Checking configuration files..."
if [ -f "./examples/blog_posts/gepa/configs/banking77_gepa_local.toml" ]; then
    echo "   ✅ banking77_gepa_local.toml exists"
else
    echo "   ❌ banking77_gepa_local.toml not found"
fi

echo ""
echo "5️⃣ Checking environment variables..."
if [ -n "$GROQ_API_KEY" ]; then
    echo "   ✅ GROQ_API_KEY is set (${GROQ_API_KEY:0:10}...)"
else
    echo "   ⚠️  GROQ_API_KEY not set"
    echo "   💡 Run: export GROQ_API_KEY='gsk_...'"
fi

if [ -n "$ENVIRONMENT_API_KEY" ]; then
    echo "   ✅ ENVIRONMENT_API_KEY is set (${ENVIRONMENT_API_KEY:0:10}...)"
else
    echo "   ⚠️  ENVIRONMENT_API_KEY not set"
    echo "   💡 Run: export ENVIRONMENT_API_KEY=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
fi

if [ -n "$SYNTH_API_KEY" ]; then
    echo "   ✅ SYNTH_API_KEY is set (${SYNTH_API_KEY:0:10}...)"
else
    echo "   ⚠️  SYNTH_API_KEY not set"
    echo "   💡 Get from backend admin or .env.dev file"
fi

echo ""
echo "6️⃣ Checking services..."
if curl -s -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "   ✅ Backend is running on http://localhost:8000"
else
    echo "   ⚠️  Backend not reachable at http://localhost:8000"
    echo "   💡 Start the backend before running GEPA"
fi

if curl -s -f http://127.0.0.1:8102/health > /dev/null 2>&1; then
    echo "   ✅ Task app is running on http://127.0.0.1:8102"
else
    echo "   ⚠️  Task app not running on http://127.0.0.1:8102"
    echo "   💡 Run: ./examples/blog_posts/gepa/deploy_banking77_task_app.sh"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To run Banking77 GEPA:"
echo ""
echo "  1. Install dependencies:"
echo "     uv pip install -e ."
echo ""
echo "  2. Set environment variables:"
echo "     export GROQ_API_KEY='gsk_...'"
echo "     export SYNTH_API_KEY='your-backend-key'"
echo "     export ENVIRONMENT_API_KEY=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo ""
echo "  3. Start task app (Terminal 1):"
echo "     ./examples/blog_posts/gepa/deploy_banking77_task_app.sh"
echo ""
echo "  4. Run GEPA (Terminal 2):"
echo "     ./examples/blog_posts/gepa/run_gepa_banking77.sh"
echo ""
echo "✅ Setup verification complete!"

