#!/usr/bin/env python3
"""Test script to verify Banking77 adapter structure without API calls."""

import sys
from pathlib import Path

# Add langprobe to path
_script_dir = Path(__file__).resolve().parent
_langprobe_dir = _script_dir.parent.parent
if str(_langprobe_dir) not in sys.path:
    sys.path.insert(0, str(_langprobe_dir))

from dspy_banking77_adapter import (
    load_banking77_dataset,
    get_available_intents,
    create_dspy_examples,
    banking77_metric,
    Banking77Classifier,
)

def test_dataset_loading():
    """Test dataset loading."""
    print("Testing dataset loading...")
    examples = load_banking77_dataset(split="train")
    print(f"✓ Loaded {len(examples)} examples")

    if examples:
        first = examples[0]
        print(f"✓ First example keys: {list(first.keys())}")
        print(f"✓ First example query: {first['query'][:80]}...")
        print(f"✓ First example intent: {first['intent']}")
    return examples

def test_available_intents():
    """Test getting available intents."""
    print("\nTesting available intents loading...")
    intents = get_available_intents()
    print(f"✓ Loaded {len(intents)} intents")

    if intents:
        print(f"✓ First few intents: {intents[:5]}")
    return intents

def test_dspy_example_creation():
    """Test DSPy example creation."""
    print("\nTesting DSPy example creation...")
    examples = load_banking77_dataset(split="train")
    intents = get_available_intents()
    train_examples = examples[:5]  # Take first 5

    dspy_examples = create_dspy_examples(train_examples, intents)
    print(f"✓ Created {len(dspy_examples)} DSPy examples")

    if dspy_examples:
        first = dspy_examples[0]
        print(f"✓ First DSPy example has query: {hasattr(first, 'query')}")
        print(f"✓ First DSPy example has available_intents: {hasattr(first, 'available_intents')}")
        print(f"✓ First DSPy example has intent: {hasattr(first, 'intent')}")
    return dspy_examples

def test_module_creation():
    """Test module instantiation."""
    print("\nTesting module instantiation...")
    module = Banking77Classifier()
    print(f"✓ Created Banking77Classifier module")
    print(f"✓ Module has predict attribute: {hasattr(module, 'predict')}")
    return module

def test_metric_function():
    """Test metric function."""
    print("\nTesting metric function...")

    import dspy

    gold = dspy.Example(intent="card_payment_wrong_exchange_rate").with_inputs()

    # Test exact match
    score1 = banking77_metric(gold, dspy.Prediction(intent="card_payment_wrong_exchange_rate"))
    print(f"✓ Exact match score: {score1} (expected 1.0)")

    # Test match with normalization (underscores vs spaces)
    score2 = banking77_metric(gold, dspy.Prediction(intent="card payment wrong exchange rate"))
    print(f"✓ Normalized match score: {score2} (expected 1.0)")

    # Test no match
    score3 = banking77_metric(gold, dspy.Prediction(intent="different_intent"))
    print(f"✓ No match score: {score3} (expected 0.0)")

    assert score1 == 1.0, "Exact match should score 1.0"
    assert score2 == 1.0, "Normalized match should score 1.0"
    assert score3 == 0.0, "No match should score 0.0"
    print("✓ Metric function works correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Banking77 Adapter Structure Tests")
    print("=" * 60)

    try:
        test_dataset_loading()
        test_available_intents()
        test_dspy_example_creation()
        test_module_creation()
        test_metric_function()

        print("\n" + "=" * 60)
        print("✅ All structure tests passed!")
        print("=" * 60)
        print("\nNote: Full optimization tests require GROQ_API_KEY")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
