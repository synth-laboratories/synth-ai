#!/usr/bin/env python3
"""Test script to verify HotPotQA adapter structure without API calls."""

import sys
from pathlib import Path

# Add langprobe to path
_script_dir = Path(__file__).resolve().parent
_langprobe_dir = _script_dir.parent.parent
if str(_langprobe_dir) not in sys.path:
    sys.path.insert(0, str(_langprobe_dir))

from dspy_hotpotqa_adapter import (
    load_hotpotqa_dataset,
    create_dspy_examples,
    hotpotqa_metric,
    HotpotQAAnswerer,
)

def test_dataset_loading():
    """Test dataset loading."""
    print("Testing dataset loading...")
    examples = load_hotpotqa_dataset(split="validation")
    print(f"✓ Loaded {len(examples)} examples")

    if examples:
        first = examples[0]
        print(f"✓ First example keys: {list(first.keys())}")
        print(f"✓ First example question: {first['question'][:80]}...")
        print(f"✓ First example answer: {first['answer']}")
        print(f"✓ First example context length: {len(first['context'])} chars")
    return examples

def test_dspy_example_creation():
    """Test DSPy example creation."""
    print("\nTesting DSPy example creation...")
    examples = load_hotpotqa_dataset(split="validation")
    train_examples = examples[:5]  # Take first 5

    dspy_examples = create_dspy_examples(train_examples)
    print(f"✓ Created {len(dspy_examples)} DSPy examples")

    if dspy_examples:
        first = dspy_examples[0]
        print(f"✓ First DSPy example has question: {hasattr(first, 'question')}")
        print(f"✓ First DSPy example has context: {hasattr(first, 'context')}")
        print(f"✓ First DSPy example has answer: {hasattr(first, 'answer')}")
    return dspy_examples

def test_module_creation():
    """Test module instantiation."""
    print("\nTesting module instantiation...")
    module = HotpotQAAnswerer()
    print(f"✓ Created HotpotQAAnswerer module")
    print(f"✓ Module has predict attribute: {hasattr(module, 'predict')}")
    return module

def test_metric_function():
    """Test metric function."""
    print("\nTesting metric function...")

    import dspy

    gold = dspy.Example(answer="Barack Obama").with_inputs()

    # Test exact match
    score1 = hotpotqa_metric(gold, dspy.Prediction(answer="Barack Obama"))
    print(f"✓ Exact match score: {score1} (expected 1.0)")

    # Test partial match (substring)
    score2 = hotpotqa_metric(gold, dspy.Prediction(answer="Obama"))
    print(f"✓ Partial match score: {score2} (expected 0.5)")

    # Test no match
    score3 = hotpotqa_metric(gold, dspy.Prediction(answer="Donald Trump"))
    print(f"✓ No match score: {score3} (expected 0.0)")

    assert score1 == 1.0, "Exact match should score 1.0"
    assert score2 == 0.5, "Partial match should score 0.5"
    assert score3 == 0.0, "No match should score 0.0"
    print("✓ Metric function works correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("HotPotQA Adapter Structure Tests")
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
