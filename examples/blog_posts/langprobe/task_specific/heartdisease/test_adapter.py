#!/usr/bin/env python3
"""Test script to verify Heart Disease adapter structure without API calls."""

import sys
from pathlib import Path

# Add langprobe to path
_script_dir = Path(__file__).resolve().parent
_langprobe_dir = _script_dir.parent.parent
if str(_langprobe_dir) not in sys.path:
    sys.path.insert(0, str(_langprobe_dir))

from dspy_heartdisease_adapter import (
    load_heartdisease_dataset,
    create_dspy_examples,
    heartdisease_metric,
    HeartDiseaseClassifier,
)

def test_dataset_loading():
    """Test dataset loading."""
    print("Testing dataset loading...")
    examples = load_heartdisease_dataset(split="train")
    print(f"✓ Loaded {len(examples)} examples")

    if examples:
        first = examples[0]
        print(f"✓ First example keys: {list(first.keys())}")
        print(f"✓ First example features preview: {first['features'][:100]}...")
        print(f"✓ First example label: {first['label']}")
    return examples

def test_dspy_example_creation():
    """Test DSPy example creation."""
    print("\nTesting DSPy example creation...")
    examples = load_heartdisease_dataset(split="train")
    train_examples = examples[:5]  # Take first 5

    dspy_examples = create_dspy_examples(train_examples)
    print(f"✓ Created {len(dspy_examples)} DSPy examples")

    if dspy_examples:
        first = dspy_examples[0]
        print(f"✓ First DSPy example has features: {hasattr(first, 'features')}")
        print(f"✓ First DSPy example has classification: {hasattr(first, 'classification')}")
    return dspy_examples

def test_module_creation():
    """Test module instantiation."""
    print("\nTesting module instantiation...")
    module = HeartDiseaseClassifier()
    print(f"✓ Created HeartDiseaseClassifier module")
    print(f"✓ Module has predict attribute: {hasattr(module, 'predict')}")
    return module

def test_metric_function():
    """Test metric function."""
    print("\nTesting metric function...")

    # Create mock examples
    import dspy

    correct_pred = dspy.Example(classification="1").with_inputs()
    correct_gold = dspy.Example(classification="1").with_inputs()

    incorrect_pred = dspy.Example(classification="0").with_inputs()

    # Test correct prediction
    score1 = heartdisease_metric(correct_gold, dspy.Prediction(classification="1"))
    print(f"✓ Correct prediction score: {score1} (expected 1.0)")

    # Test incorrect prediction
    score2 = heartdisease_metric(correct_gold, dspy.Prediction(classification="0"))
    print(f"✓ Incorrect prediction score: {score2} (expected 0.0)")

    assert score1 == 1.0, "Correct prediction should score 1.0"
    assert score2 == 0.0, "Incorrect prediction should score 0.0"
    print("✓ Metric function works correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Heart Disease Adapter Structure Tests")
    print("=" * 60)

    try:
        test_dataset_loading()
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
