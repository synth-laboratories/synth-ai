#!/bin/bash
# Quick test script to verify DSPy implementations work with real API calls
# Runs each task with a very small budget (10 rollouts) to verify functionality

set -e  # Exit on error

echo "=========================================="
echo "DSPy Implementation Quick Test"
echo "=========================================="
echo ""

# Check for API keys
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ GROQ_API_KEY not set in environment"
    echo "Please set it in .env file or export it"
    exit 1
fi

echo "✓ GROQ_API_KEY is set"
echo ""

# Test Heart Disease GEPA
echo "Testing Heart Disease GEPA (budget=10)..."
cd task_specific/heartdisease
python run_dspy_gepa_heartdisease.py --rollout-budget 10
if [ $? -eq 0 ]; then
    echo "✅ Heart Disease GEPA test passed"
else
    echo "❌ Heart Disease GEPA test failed"
    exit 1
fi
echo ""

# Test Heart Disease MIPROv2
echo "Testing Heart Disease MIPROv2 (budget=10)..."
python run_dspy_miprov2_heartdisease.py --rollout-budget 10
if [ $? -eq 0 ]; then
    echo "✅ Heart Disease MIPROv2 test passed"
else
    echo "❌ Heart Disease MIPROv2 test failed"
    exit 1
fi
echo ""

# Test HotPotQA GEPA
echo "Testing HotPotQA GEPA (budget=10)..."
cd ../hotpotqa
python run_dspy_gepa_hotpotqa.py --rollout-budget 10
if [ $? -eq 0 ]; then
    echo "✅ HotPotQA GEPA test passed"
else
    echo "❌ HotPotQA GEPA test failed"
    exit 1
fi
echo ""

# Test HotPotQA MIPROv2
echo "Testing HotPotQA MIPROv2 (budget=10)..."
python run_dspy_miprov2_hotpotqa.py --rollout-budget 10
if [ $? -eq 0 ]; then
    echo "✅ HotPotQA MIPROv2 test passed"
else
    echo "❌ HotPotQA MIPROv2 test failed"
    exit 1
fi
echo ""

# Test Banking77 GEPA
echo "Testing Banking77 GEPA (budget=10)..."
cd ../banking77
python run_dspy_gepa_banking77.py --rollout-budget 10
if [ $? -eq 0 ]; then
    echo "✅ Banking77 GEPA test passed"
else
    echo "❌ Banking77 GEPA test failed"
    exit 1
fi
echo ""

# Test Banking77 MIPROv2
echo "Testing Banking77 MIPROv2 (budget=10)..."
python run_dspy_miprov2_banking77.py --rollout-budget 10
if [ $? -eq 0 ]; then
    echo "✅ Banking77 MIPROv2 test passed"
else
    echo "❌ Banking77 MIPROv2 test failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ All DSPy implementations verified!"
echo "=========================================="
