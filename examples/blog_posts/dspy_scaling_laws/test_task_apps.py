"""Test all task apps without optimization on 20 heldout examples."""

import asyncio
import json
from pathlib import Path
from typing import Any
import sys

# Add task_apps to path
task_apps_dir = Path(__file__).resolve().parents[3] / "task_apps"
if str(task_apps_dir) not in sys.path:
    sys.path.insert(0, str(task_apps_dir))

from synth_ai.task.contracts import (
    RolloutRequest,
    PolicyConfig,
    EnvConfig,
)


async def test_task_app(task_app_id: str, num_examples: int = 20, split: str = "train") -> dict[str, Any]:
    """Test a task app without optimization."""

    print(f"\n{'='*80}")
    print(f"Testing: {task_app_id}")
    print(f"{'='*80}")

    # Import and build the task app config
    if "banking77-3step" in task_app_id:
        sys.path.insert(0, str(Path(__file__).parent / "benchmarks/banking77/pipeline_3step"))
        from banking77_3step_task_app import build_config
    elif "banking77-5step" in task_app_id:
        sys.path.insert(0, str(Path(__file__).parent / "benchmarks/banking77/pipeline_5step"))
        from banking77_5step_task_app import build_config
    elif "heartdisease-3step" in task_app_id:
        sys.path.insert(0, str(Path(__file__).parent / "benchmarks/heartdisease/pipeline_3step"))
        from heartdisease_3step_task_app import build_config
    elif "heartdisease-5step" in task_app_id:
        sys.path.insert(0, str(Path(__file__).parent / "benchmarks/heartdisease/pipeline_5step"))
        from heartdisease_5step_task_app import build_config
    elif "hotpotqa-3step" in task_app_id:
        sys.path.insert(0, str(Path(__file__).parent / "benchmarks/hotpotqa/pipeline_3step"))
        from hotpotqa_3step_task_app import build_config
    elif "hotpotqa-5step" in task_app_id:
        sys.path.insert(0, str(Path(__file__).parent / "benchmarks/hotpotqa/pipeline_5step"))
        from hotpotqa_5step_task_app import build_config
    elif "banking77-pipeline" in task_app_id or "banking77-2step" in task_app_id:
        from banking77_pipeline.banking77_pipeline_task_app import build_config
    elif "banking77" in task_app_id:
        from banking77.banking77_task_app import build_config
    elif "heartdisease" in task_app_id:
        from other_langprobe_benchmarks.heartdisease_task_app import build_config
    elif "hotpotqa" in task_app_id:
        from gepa_benchmarks.hotpotqa_task_app import build_config
    else:
        raise ValueError(f"Unknown task app: {task_app_id}")

    config = build_config()

    # Create a mock FastAPI app with state
    from fastapi import FastAPI, Request
    app = FastAPI()
    for key, value in config.app_state.items():
        setattr(app.state, key, value)

    # Test on heldout examples (seeds 1000-1019)
    test_seeds = list(range(1000, 1000 + num_examples))

    results = []
    correct_count = 0

    for i, seed in enumerate(test_seeds):
        # Create rollout request
        request = RolloutRequest(
            run_id=f"test_{task_app_id}_{seed}",
            policy=PolicyConfig(
                policy_id="baseline",
                policy_name="baseline",
                config={
                    "model": "gpt-4o-mini",
                    "inference_url": "https://api.openai.com/v1/chat/completions",
                }
            ),
            env=EnvConfig(
                env_name=config.app_id,
                seed=seed,
                config={"split": split}
            ),
        )

        # Create mock FastAPI request
        mock_request = type('MockRequest', (), {
            'app': app,
            'headers': {},
        })()

        try:
            response = await config.rollout(request, mock_request)

            # Extract reward/correctness
            reward = response.metrics.mean_return if response.metrics else 0.0
            details = response.metrics.details if response.metrics else {}

            # Different benchmarks have different correctness keys
            is_correct = (
                details.get("correct", False) or
                details.get("label_correct", False) or
                details.get("answer_correct", False) or
                (reward > 0.5)  # Fallback for reward-based
            )

            if is_correct:
                correct_count += 1

            results.append({
                "seed": seed,
                "reward": reward,
                "correct": is_correct,
                "details": details,
            })

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{num_examples} | Accuracy so far: {correct_count}/{i+1} ({100*correct_count/(i+1):.1f}%)")

        except Exception as e:
            print(f"  ❌ Error on seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "seed": seed,
                "reward": 0.0,
                "correct": False,
                "error": str(e),
            })

    accuracy = correct_count / num_examples if num_examples > 0 else 0.0

    print(f"\n📊 Results for {task_app_id}:")
    print(f"  Accuracy: {correct_count}/{num_examples} ({100*accuracy:.1f}%)")
    print(f"  Mean Reward: {sum(r['reward'] for r in results) / len(results):.3f}")

    return {
        "task_app_id": task_app_id,
        "num_examples": num_examples,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "mean_reward": sum(r["reward"] for r in results) / len(results) if results else 0.0,
        "results": results,
    }


async def main():
    """Test all task apps."""

    print("🧪 Testing all task apps without optimization\n")
    print("Testing on heldout examples (seeds 1000-1019)")
    print("Using baseline prompts with gpt-4o-mini\n")

    # Define all task apps to test
    task_apps = [
        # Banking77
        ("banking77", "Banking77 1-step"),
        ("banking77-pipeline", "Banking77 2-step"),
        ("banking77-3step", "Banking77 3-step"),
        ("banking77-5step", "Banking77 5-step"),

        # HeartDisease
        ("heartdisease", "HeartDisease 1-step"),
        ("heartdisease-3step", "HeartDisease 3-step"),
        ("heartdisease-5step", "HeartDisease 5-step"),

        # HotpotQA
        ("hotpotqa", "HotpotQA 1-step"),
        ("hotpotqa-3step", "HotpotQA 3-step"),
        ("hotpotqa-5step", "HotpotQA 5-step"),
    ]

    all_results = []

    for task_app_id, task_app_name in task_apps:
        try:
            result = await test_task_app(task_app_id, num_examples=20)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to test {task_app_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    for result in all_results:
        task_id = result["task_app_id"]
        accuracy = result["accuracy"]
        correct = result["correct_count"]
        total = result["num_examples"]

        status = "✅" if accuracy > 0.1 else "⚠️"  # At least 10% correct
        print(f"{status} {task_id:30s} {correct:2d}/{total} ({100*accuracy:5.1f}%)")

    # Save results
    output_file = Path(__file__).parent / "results" / "baseline_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📁 Detailed results saved to: {output_file}")

    # Check if any failed completely
    failed = [r for r in all_results if r["accuracy"] == 0.0]
    if failed:
        print(f"\n⚠️  WARNING: {len(failed)} task apps got 0% accuracy:")
        for r in failed:
            print(f"   - {r['task_app_id']}")
        print("\nThese may have issues and should be debugged before optimization.")
    else:
        print("\n✅ All task apps are working! Ready for optimization.")


if __name__ == "__main__":
    asyncio.run(main())
