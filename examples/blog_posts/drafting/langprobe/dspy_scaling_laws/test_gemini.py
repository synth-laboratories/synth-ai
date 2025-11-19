#!/usr/bin/env python3
"""Quick test script to verify Gemini model works with DSPy GEPA."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hotpotqa.dspy_hotpotqa_scaling_adapter import run_dspy_gepa_hotpotqa_scaling


async def main():
    """Run a quick test with Gemini model."""
    print("🧪 Testing DSPy GEPA with gemini-2.5-flash-lite")
    print("=" * 80)
    
    # Run with minimal budget for quick test
    results = await run_dspy_gepa_hotpotqa_scaling(
        num_calls=1,  # Start with 1 call for fastest test
        rollout_budget=10,  # Very small budget for quick test
        model="gemini-2.5-flash-lite",  # Will be converted to gemini/gemini-2.5-flash-lite
        train_seeds=list(range(5)),  # Just 5 training examples
        val_seeds=list(range(50, 55)),  # Just 5 validation examples
    )
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print(f"Results: {results}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

