#!/usr/bin/env python3
"""
Test script to verify the data splits are working correctly.

Usage:
    uv run python test_data_splits.py
"""

from __future__ import annotations

from data_splits import (
    Banking77SplitDataset,
    format_available_intents,
    get_split_intents,
    get_split_size,
    print_split_info,
)


def test_split_sizes():
    """Verify split sizes are correct."""
    print("Testing split sizes...")
    assert get_split_size(1) == 2, f"Split 1 should have 2 intents, got {get_split_size(1)}"
    assert get_split_size(2) == 7, f"Split 2 should have 7 intents, got {get_split_size(2)}"
    assert get_split_size(3) == 27, f"Split 3 should have 27 intents, got {get_split_size(3)}"
    assert get_split_size(4) == 77, f"Split 4 should have 77 intents, got {get_split_size(4)}"
    print("  ✓ Split sizes correct")


def test_split_supersets():
    """Verify each split is a superset of the previous."""
    print("Testing split supersets...")
    for i in range(1, 4):
        current = set(get_split_intents(i))
        next_split = set(get_split_intents(i + 1))
        assert current.issubset(next_split), f"Split {i} should be a subset of Split {i+1}"
    print("  ✓ Each split is a superset of the previous")


def test_dataset_loading():
    """Verify the dataset loads correctly for each split."""
    print("Testing dataset loading...")
    dataset = Banking77SplitDataset()
    dataset.ensure_ready(["train", "test"])
    
    for split_num in [1, 2, 3, 4]:
        train_size = dataset.size("train", split_num)
        test_size = dataset.size("test", split_num)
        
        assert train_size > 0, f"Split {split_num} train should have samples"
        assert test_size > 0, f"Split {split_num} test should have samples"
        
        # Verify sample is from correct intents
        sample = dataset.sample(data_split="train", intent_split=split_num, index=0)
        allowed_intents = set(get_split_intents(split_num))
        assert sample["label"] in allowed_intents, \
            f"Sample label {sample['label']} not in split {split_num} intents"
        
        print(f"  Split {split_num}: train={train_size}, test={test_size}")
    
    print("  ✓ Dataset loading correct")


def test_format_intents():
    """Verify intent formatting works."""
    print("Testing intent formatting...")
    intents = get_split_intents(1)
    formatted = format_available_intents(intents)
    
    assert "1. card_arrival" in formatted, "First intent should be numbered"
    assert "2. lost_or_stolen_card" in formatted, "Second intent should be numbered"
    print("  ✓ Intent formatting correct")


def main():
    print("="*60)
    print("Banking77 Data Splits Test Suite")
    print("="*60)
    
    test_split_sizes()
    test_split_supersets()
    test_dataset_loading()
    test_format_intents()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    
    # Print detailed split info
    print("\n")
    print_split_info()


if __name__ == "__main__":
    main()
