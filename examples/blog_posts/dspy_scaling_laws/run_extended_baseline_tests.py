"""Run extended baseline tests on all task apps with multiple models."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any
import sys
from datetime import datetime

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


# Model configurations to test
MODEL_CONFIGS = {
    "groq-20b": {
        "model": "openai/gpt-oss-20b",
        "provider": "groq",
        "inference_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
}


async def test_task_app_with_model(
    config_factory,
    task_app_name: str,
    model_name: str,
    model_config: dict[str, Any],
    num_examples: int = 100,
    split: str = "train",
) -> dict[str, Any]:
    """Test a task app with a specific model configuration."""

    print(f"\n{'='*80}")
    print(f"Testing: {task_app_name} | Model: {model_name}")
    print(f"{'='*80}")

    # Build config - EXACT PATTERN FROM WORKING run_baseline_tests.py
    config = config_factory()

    # Create FastAPI app - EXACT PATTERN FROM WORKING run_baseline_tests.py
    from fastapi import FastAPI
    app = FastAPI()
    for key, value in config.app_state.items():
        setattr(app.state, key, value)

    # Initialize HTTP client for Banking77 tasks
    import httpx
    if not hasattr(app.state, 'http_client') or app.state.http_client is None:
        app.state.http_client = httpx.AsyncClient()

    # Get API key from environment
    api_key = os.getenv(model_config["api_key_env"], "")
    if not api_key:
        print(f"⚠️  WARNING: {model_config['api_key_env']} not set!")
        return {
            "task_app": task_app_name,
            "model": model_name,
            "num_examples": 0,
            "accuracy": 0.0,
            "error": f"Missing {model_config['api_key_env']}",
        }

    # EXACT PATTERN FROM WORKING run_baseline_tests.py
    class MockRequest:
        def __init__(self):
            self.app = app
            self.headers = {}
            if api_key:
                self.headers["X-API-Key"] = api_key

    mock_request = MockRequest()

    # Test on heldout seeds (1000-1099 for 100 examples)
    test_seeds = list(range(1000, 1000 + num_examples))

    results = []
    correct_count = 0
    total_reward = 0.0
    error_count = 0

    for i, seed in enumerate(test_seeds):
        # Create rollout request
        request = RolloutRequest(
            run_id=f"extended_test_{model_name}_{seed}",
            policy=RolloutPolicySpec(
                policy_id=f"baseline_{model_name}",
                policy_name=f"baseline_test_{model_name}",
                config={
                    "model": model_config["model"],
                    "provider": model_config["provider"],
                    "inference_url": model_config["inference_url"],
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

            # Progress updates every 10 examples
            if (i + 1) % 10 == 0:
                acc = 100 * correct_count / (i + 1)
                avg_reward = total_reward / (i + 1)
                print(f"  [{i+1}/{num_examples}] Accuracy: {correct_count}/{i+1} ({acc:.1f}%) | Avg Reward: {avg_reward:.3f}")

        except Exception as e:
            error_count += 1
            print(f"  ❌ Error on seed {seed}: {e}")
            results.append({
                "seed": seed,
                "reward": 0.0,
                "correct": False,
                "error": str(e),
            })

            # Stop if too many errors
            if error_count > 5:
                print(f"  ⚠️  Too many errors ({error_count}), stopping this test")
                break

    accuracy = correct_count / len(results) if results else 0.0
    avg_reward = total_reward / len(results) if results else 0.0

    print(f"\n📊 Results for {task_app_name} | {model_name}:")
    print(f"  Accuracy: {correct_count}/{len(results)} ({100*accuracy:.1f}%)")
    print(f"  Avg Reward: {avg_reward:.3f}")
    print(f"  Errors: {error_count}")

    return {
        "task_app": task_app_name,
        "model": model_name,
        "model_config": model_config["model"],
        "num_examples": len(results),
        "num_completed": len(results),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "avg_reward": avg_reward,
        "error_count": error_count,
        "results": results,
    }


async def main():
    """Run extended baseline tests on all task apps with multiple models."""

    # Enable direct provider URLs for testing
    os.environ["ALLOW_DIRECT_PROVIDER_URLS"] = "1"

    print("🧪 EXTENDED BASELINE TESTING")
    print("=" * 80)
    print("Testing Configuration:")
    print("  - Test Set: Seeds 1000-1099 (100 heldout examples)")
    print("  - Models:")
    for name, cfg in MODEL_CONFIGS.items():
        print(f"    - {name}: {cfg['model']} ({cfg['provider']})")
    print("=" * 80)

    # Check for API keys
    missing_keys = []
    for name, cfg in MODEL_CONFIGS.items():
        if not os.getenv(cfg["api_key_env"]):
            missing_keys.append(cfg["api_key_env"])
            print(f"⚠️  Warning: {cfg['api_key_env']} not set for {name}")

    if missing_keys:
        print(f"\n⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("Tests will be skipped for models with missing keys.\n")

    # Import task apps - ONLY 1-STEP TASKS (multi-step have architectural issues)
    task_apps_to_test = []

    # HeartDisease 1-step
    try:
        from other_langprobe_benchmarks.heartdisease_task_app import build_config as heartdisease_1step_config
        task_apps_to_test.append((heartdisease_1step_config, "HeartDisease 1-step"))
    except Exception as e:
        print(f"⚠️  Could not import HeartDisease 1-step: {e}")

    # HotpotQA 1-step
    try:
        from gepa_benchmarks.hotpotqa_task_app import build_config as hotpotqa_1step_config
        task_apps_to_test.append((hotpotqa_1step_config, "HotpotQA 1-step"))
    except Exception as e:
        print(f"⚠️  Could not import HotpotQA 1-step: {e}")

    print(f"\n✅ Loaded {len(task_apps_to_test)} task apps to test")
    print(f"📊 Total tests to run: {len(task_apps_to_test)} × {len(MODEL_CONFIGS)} = {len(task_apps_to_test) * len(MODEL_CONFIGS)}\n")

    # Run tests for each model
    all_results = {}
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\n{'#'*80}")
        print(f"# TESTING WITH MODEL: {model_name} ({model_config['model']})")
        print(f"{'#'*80}\n")

        # Skip if API key missing
        if not os.getenv(model_config["api_key_env"]):
            print(f"⚠️  Skipping {model_name} - missing {model_config['api_key_env']}\n")
            continue

        model_results = []
        for config_factory, name in task_apps_to_test:
            try:
                result = await test_task_app_with_model(
                    config_factory, name, model_name, model_config, num_examples=100
                )
                model_results.append(result)
            except Exception as e:
                print(f"\n❌ Failed to test {name} with {model_name}: {e}")
                import traceback
                traceback.print_exc()

        all_results[model_name] = model_results

    # Generate summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE SUMMARY - Extended Baseline Performance")
    print("="*80)

    for model_name in MODEL_CONFIGS.keys():
        if model_name not in all_results:
            continue

        print(f"\n{model_name.upper()} ({MODEL_CONFIGS[model_name]['model']}):")
        print("-" * 80)

        for result in all_results[model_name]:
            name = result["task_app"]
            acc = result["accuracy"]
            correct = result["correct_count"]
            total = result["num_completed"]
            reward = result["avg_reward"]

            # Status indicators
            if acc >= 0.3:
                status = "✅ GOOD"
            elif acc >= 0.1:
                status = "⚠️  OK"
            else:
                status = "❌ LOW"

            print(f"{status:10s} {name:25s} {correct:3d}/{total:3d} ({100*acc:5.1f}%) | Reward: {reward:.3f}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"extended_baseline_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("📈 COMPARISON ACROSS MODELS")
    print("="*80)

    # Create comparison table by task app
    task_app_names = [name for _, name in task_apps_to_test]

    print(f"\n{'Task App':<30} ", end="")
    for model_name in MODEL_CONFIGS.keys():
        if model_name in all_results:
            print(f"{model_name:<15}", end="")
    print()
    print("-" * (30 + 15 * len(all_results)))

    for task_name in task_app_names:
        print(f"{task_name:<30} ", end="")
        for model_name in MODEL_CONFIGS.keys():
            if model_name not in all_results:
                continue
            # Find result for this task
            task_result = next((r for r in all_results[model_name] if r["task_app"] == task_name), None)
            if task_result:
                acc = task_result["accuracy"]
                print(f"{100*acc:5.1f}%         ", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    print("\n✅ Extended baseline testing complete!")
    print(f"📝 Tested {len(task_apps_to_test)} task apps with {len(all_results)} models")
    print(f"📊 Total of {sum(len(results) for results in all_results.values())} test runs completed")


if __name__ == "__main__":
    asyncio.run(main())
