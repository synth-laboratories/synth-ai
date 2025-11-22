"""Quick test to verify the scaling experiment setup works."""

import asyncio
import os
from pathlib import Path
from run_scaling_experiment import run_single_experiment


async def main():
    """Run a quick test experiment with reduced budget."""

    print("🧪 Testing DSPy Scaling Experiment Setup\n")

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment")
        print("   Please set it with: export GROQ_API_KEY='your_key'")
        return

    print("✅ GROQ_API_KEY found")

    # Run a quick test with Banking77 1-step GEPA
    print("\n📊 Running test experiment:")
    print("   Benchmark: Banking77")
    print("   Pipeline: 1-step (baseline)")
    print("   Optimizer: GEPA")
    print("   Budget: 20 rollouts (reduced for testing)")

    test_dir = Path(__file__).parent / "results" / "test"

    try:
        result = await run_single_experiment(
            benchmark="banking77",
            num_steps=1,
            optimizer="gepa",
            output_dir=test_dir,
            rollout_budget=20,  # Reduced for quick test
        )

        print("\n✅ Test completed successfully!")
        print(f"\n📊 Results:")
        print(f"   Baseline: {result['baseline_score']:.4f}")
        print(f"   Final: {result['final_score']:.4f}")
        print(f"   Improvement: {result['improvement']:+.4f}")
        print(f"\n📁 Results saved to: {test_dir}")

        print("\n🎉 Setup verified! Ready to run full experiments.")
        print("\nNext steps:")
        print("  1. Run full experiments: python run_scaling_experiment.py")
        print("  2. Analyze results: python analyze_results.py")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Check GROQ_API_KEY is valid")
        print("  - Ensure dependencies are installed: pip install dspy-ai datasets")
        print("  - Check internet connection")


if __name__ == "__main__":
    asyncio.run(main())
