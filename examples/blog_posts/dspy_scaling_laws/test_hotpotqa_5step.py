"""Test HotpotQA 5-step with openai/gpt-oss-20b."""

import asyncio
import os
import sys
from pathlib import Path

# Add paths
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

task_apps_dir = repo_root / "examples" / "task_apps"
if str(task_apps_dir) not in sys.path:
    sys.path.insert(0, str(task_apps_dir))

from synth_ai.task.contracts import (
    RolloutRequest,
    RolloutPolicySpec,
    RolloutEnvSpec,
    RolloutMode,
)


async def test_hotpotqa_5step():
    """Test HotpotQA 5-step with 3 examples."""

    # Enable direct provider URLs
    os.environ["ALLOW_DIRECT_PROVIDER_URLS"] = "1"

    # Import task app
    from hotpotqa_5step.hotpotqa_5step_task_app import build_config

    # Build config
    config = build_config()

    # Create mock FastAPI app
    from fastapi import FastAPI
    import httpx

    app = FastAPI()
    for key, value in config.app_state.items():
        setattr(app.state, key, value)

    # Run startup hooks
    for hook in config.startup_hooks:
        await hook(app)

    # Get API key
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not set")
        return

    class MockRequest:
        def __init__(self):
            self.app = app
            self.headers = {"X-API-Key": groq_api_key}

    mock_request = MockRequest()

    # Test 3 examples
    print(f"\n{'='*80}")
    print("Testing HotpotQA 5-step with openai/gpt-oss-20b (3 examples)")
    print(f"{'='*80}\n")

    correct = 0
    for i, seed in enumerate(range(1000, 1003)):
        request = RolloutRequest(
            run_id=f"test_{seed}",
            policy=RolloutPolicySpec(
                policy_id="test",
                policy_name="test",
                config={
                    "model": "openai/gpt-oss-20b",
                    "provider": "groq",
                    "inference_url": "https://api.groq.com/openai/v1",
                }
            ),
            env=RolloutEnvSpec(
                env_name=config.app_id,
                seed=seed,
                config={"split": "train"}
            ),
            ops=[],
            mode=RolloutMode.EVAL,
        )

        try:
            response = await config.rollout(request, mock_request)
            reward = response.metrics.mean_return if response.metrics else 0.0
            details = response.metrics.details if response.metrics else {}
            answer_correct = details.get("answer_correct", False)

            if answer_correct:
                correct += 1

            print(f"  [{i+1}/3] Seed {seed}: {'✅' if answer_correct else '❌'} (reward={reward:.2f})")
        except Exception as e:
            print(f"  [{i+1}/3] Seed {seed}: ❌ ERROR: {str(e)[:200]}")
            import traceback
            traceback.print_exc()

    # Run shutdown hooks
    for hook in config.shutdown_hooks:
        await hook(app)

    print(f"\n{'='*80}")
    print(f"Result: {correct}/3 correct ({100*correct/3:.1f}%)")
    print(f"{'='*80}\n")

    if correct > 0:
        print("✅ HotpotQA 5-step is WORKING!")
    else:
        print("❌ HotpotQA 5-step still FAILING - need to investigate errors above")


if __name__ == "__main__":
    asyncio.run(test_hotpotqa_5step())
