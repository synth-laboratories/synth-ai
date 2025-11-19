#!/usr/bin/env python3
"""Simple test to verify Gemini model works with DSPy."""

import os
import dspy
from dotenv import load_dotenv

load_dotenv()

# Test Gemini model configuration
model = "google/gemini-2.5-flash-lite"
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in environment")
    exit(1)

print(f"🧪 Testing DSPy with {model}")
print(f"📝 API Key: {api_key[:10]}...")

try:
    # Try different ways to configure DSPy LM
    print("\n1. Testing with api_key parameter...")
    lm = dspy.LM(model, api_key=api_key)
    
    # Test a simple call
    print("2. Making a test call...")
    result = lm("What is 2+2? Answer in one word.")
    print(f"✅ Result: {result}")
    
    # Test with a signature
    print("\n3. Testing with a signature...")
    class SimpleQA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    
    predictor = dspy.ChainOfThought(SimpleQA)
    with dspy.context(lm=lm):
        pred = predictor(question="What is 2+2?")
        print(f"✅ Prediction: {pred}")
        print(f"   Answer field: {pred.answer}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Try alternative: set env var
    print("\n4. Trying with GEMINI_API_KEY env var set...")
    os.environ["GEMINI_API_KEY"] = api_key
    try:
        lm2 = dspy.LM(model)
        result2 = lm2("What is 2+2?")
        print(f"✅ Result with env var: {result2}")
    except Exception as e2:
        print(f"❌ Still failed: {e2}")
        import traceback
        traceback.print_exc()


