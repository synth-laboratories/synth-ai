"""Run baseline tests on all task apps without optimization."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any
import sys

# Ensure we can import synth_ai and task_apps
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Add task_apps to path
task_apps_dir = repo_root / "examples" / "task_apps"
if str(task_apps_dir) not in sys.path:
    sys.path.insert(0, str(task_apps_dir))

from synth_ai.task.contracts import (
    RolloutRequest,
    RolloutPolicySpec,
    RolloutEnvSpec,
    RolloutMode,
)


async def test_task_app_baseline(
    config_factory,
    task_app_name: str,
    num_examples: int = 100,
    split: str = "train",
) -> dict[str, Any]:
    """Test a task app with baseline prompt (no optimization)."""

    print(f"\n{'='*80}")
    print(f"Testing: {task_app_name}")
    print(f"{'='*80}")

    # Build config
    config = config_factory()

    # Create mock FastAPI app
    from fastapi import FastAPI
    app = FastAPI()
    for key, value in config.app_state.items():
        setattr(app.state, key, value)

    # Initialize HTTP client for Banking77 tasks
    import httpx
    if not hasattr(app.state, 'http_client') or app.state.http_client is None:
        app.state.http_client = httpx.AsyncClient()

    # Create mock request with API key
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    class MockRequest:
        def __init__(self):
            self.app = app
            self.headers = {}
            if groq_api_key:
                self.headers["X-API-Key"] = groq_api_key

    mock_request = MockRequest()

    # Test on heldout seeds (1000-1019)
    test_seeds = list(range(1000, 1000 + num_examples))

    results = []
    correct_count = 0
    total_reward = 0.0

    for i, seed in enumerate(test_seeds):
        # Create rollout request
        request = RolloutRequest(
            run_id=f"baseline_test_{seed}",
            policy=RolloutPolicySpec(
                policy_id="baseline",
                policy_name="baseline_test",
                config={
                    "model": "llama-3.1-8b-instant",
                    "provider": "groq",
                    "inference_url": "https://api.groq.com/openai/v1",
                }
            ),
            env=RolloutEnvSpec(
                env_name=config.app_id,
                seed=seed,
                config={"split": split}
            ),
            ops=[],
            mode=RolloutMode.EVAL,
        )

        try:
            response = await config.rollout(request, mock_request)

            # Extract metrics
            reward = response.metrics.mean_return if response.metrics else 0.0
            details = response.metrics.details if response.metrics else {}

            # Check correctness (different keys for different tasks)
            is_correct = (
                details.get("correct", False) or
                details.get("label_correct", False) or
                details.get("answer_correct", False) or
                (reward >= 0.5)  # Fallback
            )

            if is_correct:
                correct_count += 1

            total_reward += reward

            results.append({
                "seed": seed,
                "reward": float(reward),
                "correct": bool(is_correct),
                "details": details,
            })

            # Progress
            if (i + 1) % 5 == 0 or (i + 1) == num_examples:
                acc = 100 * correct_count / (i + 1)
                avg_reward = total_reward / (i + 1)
                print(f"  [{i+1}/{num_examples}] Accuracy: {correct_count}/{i+1} ({acc:.1f}%) | Avg Reward: {avg_reward:.3f}")

        except Exception as e:
            print(f"  ❌ Error on seed {seed}: {e}")
            results.append({
                "seed": seed,
                "reward": 0.0,
                "correct": False,
                "error": str(e),
            })

    accuracy = correct_count / num_examples if num_examples > 0 else 0.0
    avg_reward = total_reward / num_examples if num_examples > 0 else 0.0

    print(f"\n📊 Final Results:")
    print(f"  Accuracy: {correct_count}/{num_examples} ({100*accuracy:.1f}%)")
    print(f"  Avg Reward: {avg_reward:.3f}")

    return {
        "task_app": task_app_name,
        "num_examples": num_examples,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "avg_reward": avg_reward,
        "results": results,
    }


async def main():
    """Run baseline tests on all task apps."""

    # Enable direct provider URLs for testing
    os.environ["ALLOW_DIRECT_PROVIDER_URLS"] = "1"

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not set. Set it with:")
        print("   export GROQ_API_KEY=your_key_here")
        print("\nProceeding anyway - will fail if API key is required.\n")

    print("🧪 BASELINE TESTING - No Optimization")
    print("Testing on heldout examples (seeds 1000-1099)")
    print("Using openai/gpt-oss-20b (Groq) with default prompts\n")

    # Import task apps
    task_apps_to_test = []

    # Banking77 (existing baseline)
    try:
        from banking77.banking77_task_app import build_config as banking77_1step_config
        task_apps_to_test.append((banking77_1step_config, "Banking77 1-step"))
    except Exception as e:
        print(f"⚠️  Could not import Banking77 1-step: {e}")

    # Banking77 2-step (existing)
    try:
        from banking77_pipeline.banking77_pipeline_task_app import build_config as banking77_2step_config
        task_apps_to_test.append((banking77_2step_config, "Banking77 2-step"))
    except Exception as e:
        print(f"⚠️  Could not import Banking77 2-step: {e}")

    # Banking77 3-step (new)
    try:
        from banking77_3step.banking77_3step_task_app import build_config as banking77_3step_config
        task_apps_to_test.append((banking77_3step_config, "Banking77 3-step"))
    except Exception as e:
        print(f"⚠️  Could not import Banking77 3-step: {e}")

    # Banking77 5-step (new)
    try:
        from banking77_5step.banking77_5step_task_app import build_config as banking77_5step_config
        task_apps_to_test.append((banking77_5step_config, "Banking77 5-step"))
    except Exception as e:
        print(f"⚠️  Could not import Banking77 5-step: {e}")

    # HeartDisease 1-step (existing)
    try:
        from other_langprobe_benchmarks.heartdisease_task_app import build_config as heartdisease_1step_config
        task_apps_to_test.append((heartdisease_1step_config, "HeartDisease 1-step"))
    except Exception as e:
        print(f"⚠️  Could not import HeartDisease 1-step: {e}")

    # HeartDisease 3-step (new)
    try:
        from heartdisease_3step.heartdisease_3step_task_app import build_config as heartdisease_3step_config
        task_apps_to_test.append((heartdisease_3step_config, "HeartDisease 3-step"))
    except Exception as e:
        print(f"⚠️  Could not import HeartDisease 3-step: {e}")

    # HeartDisease 5-step (new)
    try:
        from heartdisease_5step.heartdisease_5step_task_app import build_config as heartdisease_5step_config
        task_apps_to_test.append((heartdisease_5step_config, "HeartDisease 5-step"))
    except Exception as e:
        print(f"⚠️  Could not import HeartDisease 5-step: {e}")

    # HotpotQA 1-step (existing)
    try:
        from gepa_benchmarks.hotpotqa_task_app import build_config as hotpotqa_1step_config
        task_apps_to_test.append((hotpotqa_1step_config, "HotpotQA 1-step"))
    except Exception as e:
        print(f"⚠️  Could not import HotpotQA 1-step: {e}")

    # HotpotQA 3-step (new)
    try:
        from hotpotqa_3step.hotpotqa_3step_task_app import build_config as hotpotqa_3step_config
        task_apps_to_test.append((hotpotqa_3step_config, "HotpotQA 3-step"))
    except Exception as e:
        print(f"⚠️  Could not import HotpotQA 3-step: {e}")

    # HotpotQA 5-step (new)
    try:
        from hotpotqa_5step.hotpotqa_5step_task_app import build_config as hotpotqa_5step_config
        task_apps_to_test.append((hotpotqa_5step_config, "HotpotQA 5-step"))
    except Exception as e:
        print(f"⚠️  Could not import HotpotQA 5-step: {e}")

    print(f"\n✅ Loaded {len(task_apps_to_test)} task apps to test\n")

    # Run tests
    all_results = []
    for config_factory, name in task_apps_to_test:
        try:
            result = await test_task_app_baseline(config_factory, name, num_examples=100)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY - Baseline Performance")
    print("="*80)

    for result in all_results:
        name = result["task_app"]
        acc = result["accuracy"]
        correct = result["correct_count"]
        total = result["num_examples"]
        reward = result["avg_reward"]

        # Status indicators
        if acc >= 0.3:
            status = "✅ GOOD"
        elif acc >= 0.1:
            status = "⚠️  OK"
        else:
            status = "❌ LOW"

        print(f"{status:10s} {name:25s} {correct:2d}/{total} ({100*acc:5.1f}%) | Reward: {reward:.3f}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_test_results.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}")

    # Recommendations
    low_accuracy = [r for r in all_results if r["accuracy"] < 0.1]
    if low_accuracy:
        print(f"\n⚠️  WARNING: {len(low_accuracy)} task apps have <10% accuracy:")
        for r in low_accuracy:
            print(f"   - {r['task_app']}")
        print("\n   These should be debugged before running optimization.")
    else:
        print("\n✅ All task apps show reasonable baseline performance!")
        print("   Ready to proceed with GEPA/MIPRO optimization.")

    print("\n📝 Next steps:")
    print("   1. Review any low-accuracy task apps")
    print("   2. If all look good, run full optimization experiments")
    print("   3. Use analyze_results.py to visualize scaling curves")


if __name__ == "__main__":
    asyncio.run(main())
