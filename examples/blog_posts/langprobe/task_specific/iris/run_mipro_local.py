#!/usr/bin/env python3
"""Run MIPRO locally on Iris for rapid iteration.

This script bypasses the adapter layer and uses the backend optimizer directly,
allowing rapid iteration on fixing step.info issues.

Usage:
    python run_mipro_local.py --task-app-url http://127.0.0.1:8115 --rollout-budget 20
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose HTTP logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Add backend to path
monorepo_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "monorepo"
backend_path = monorepo_root / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))
else:
    # Fallback: try environment variable
    backend_path = os.getenv("MONOREPO_BACKEND_PATH")
    if backend_path:
        sys.path.insert(0, backend_path)
    else:
        print("ERROR: Could not find monorepo backend. Set MONOREPO_BACKEND_PATH env var.")
        sys.exit(1)

# Add synth-ai to path
synth_ai_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "synth-ai"
if synth_ai_root.exists():
    sys.path.insert(0, str(synth_ai_root))
else:
    synth_ai_root = os.getenv("SYNTH_AI_ROOT")
    if synth_ai_root:
        sys.path.insert(0, synth_ai_root)

from app.routes.prompt_learning.algorithm.mipro import (
    MIPROOptimizer,
    MIPROConfig,
    MIPROSeedConfig,
    MIPROModuleConfig,
    MIPROStageConfig,
    MIPROMetaConfig,
)
from app.routes.prompt_learning.core.runtime import LocalRuntime
from app.routes.prompt_learning.core.patterns import PromptPattern, MessagePattern


class NoOpEmitter:
    """No-op event emitter for local testing."""
    async def append_event(self, **_kwargs) -> None:
        pass

    async def append_metric(self, **_kwargs) -> None:
        pass


def create_iris_mipro_config(
    task_app_url: str,
    rollout_budget: int = 20,
    bootstrap_seeds: Optional[list[int]] = None,
    online_seeds: Optional[list[int]] = None,
    test_seeds: Optional[list[int]] = None,
) -> MIPROConfig:
    """Create MIPRO config for Iris classification."""
    
    # Auto-scale seeds based on budget
    if bootstrap_seeds is None:
        # Small bootstrap set for quick iteration
        bootstrap_seeds = list(range(0, min(5, rollout_budget // 4)))
    
    if online_seeds is None:
        # Online pool: use remaining budget
        max_bootstrap = max(bootstrap_seeds) if bootstrap_seeds else -1
        online_size = min(10, rollout_budget // 2)
        online_seeds = list(range(max_bootstrap + 1, max_bootstrap + 1 + online_size))
    
    if test_seeds is None:
        # Test pool: seeds after online
        max_online = max(online_seeds) if online_seeds else max(bootstrap_seeds) if bootstrap_seeds else -1
        test_seeds = list(range(max_online + 1, max_online + 1 + 10))
    
    # Auto-scale iterations based on budget
    # Conservative: ensure we don't hit TPE errors
    num_iterations = max(2, min(5, rollout_budget // 10))
    num_evaluations_per_iteration = max(2, min(5, rollout_budget // (num_iterations * 2)))
    
    # Get API key
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
    
    # Policy config - use Groq models (gpt-oss format)
    policy_config = {
        "model": os.getenv("POLICY_MODEL", "openai/gpt-oss-20b"),  # Groq format
        "provider": "groq",
        "temperature": 0.0,
        "max_completion_tokens": 128,
        "policy_name": "iris-mipro-local",
    }
    
    # Meta model config for instruction proposal
    # Groq model names - use a model that actually exists on Groq
    # Try llama-3.3-70b-versatile or check Groq docs for available models
    meta_model_name = os.getenv("META_MODEL", "llama-3.3-70b-versatile")  # Known Groq model
    if meta_model_name.startswith("openai/"):
        meta_model_name = meta_model_name.replace("openai/", "")
    if meta_model_name.startswith("groq/"):
        meta_model_name = meta_model_name.replace("groq/", "")
    
    meta_config = MIPROMetaConfig(
        model=meta_model_name,  # Just model name, e.g., "llama-3.3-70b-versatile"
        provider="groq",
        inference_url="https://api.groq.com",  # Base URL only - AsyncGroq adds /openai/v1 automatically
        temperature=0.7,
        max_tokens=512,
    )
    
    # Module config (single-stage for Iris)
    # MIPROModuleConfig contains stages, not direct module config
    stage_config = MIPROStageConfig(
        stage_id="primary",
        max_instruction_slots=3,
        max_demo_slots=5,
    )
    
    module_config = MIPROModuleConfig(
        module_id="iris_module",
        stages=[stage_config],
    )
    
    # Seed config
    seed_config = MIPROSeedConfig(
        bootstrap=bootstrap_seeds,
        online=online_seeds,
        test=test_seeds,
        reference=[],
    )
    
    config = MIPROConfig(
        task_app_url=task_app_url,
        task_app_api_key=task_app_api_key,
        env_name="iris",
        env_config=None,
        policy_config=policy_config,
        meta=meta_config,
        modules=[module_config],  # List of modules
        seeds=seed_config,
        num_iterations=num_iterations,
        num_evaluations_per_iteration=num_evaluations_per_iteration,
        batch_size=min(5, num_evaluations_per_iteration),
        max_concurrent=5,
        few_shot_score_threshold=0.7,
        max_token_limit=4096,  # Increased for meta-model proposals
        token_counting_model="gpt-4",
        enforce_token_limit=False,  # Disable for rapid iteration
        max_spend_usd=None,
    )
    
    return config


def create_iris_pattern() -> PromptPattern:
    """Create initial prompt pattern for Iris classification."""
    return PromptPattern(
        messages=[
            MessagePattern(
                role="system",
                pattern=(
                    "You are a botany classification assistant. Based on the flower's measurements, "
                    "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
                ),
                order=0,
            ),
            MessagePattern(
                role="user",
                pattern=(
                    "Flower Measurements:\n{features}\n\n"
                    "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
                ),
                order=1,
            ),
        ],
        wildcards={"features": "REQUIRED"},
    )


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MIPRO locally on Iris")
    parser.add_argument(
        "--task-app-url",
        type=str,
        default="http://127.0.0.1:8115",
        help="Task app URL (default: http://127.0.0.1:8115)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=20,
        help="Rollout budget (default: 20)",
    )
    parser.add_argument(
        "--bootstrap-seeds",
        type=int,
        nargs="+",
        help="Bootstrap seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--online-seeds",
        type=int,
        nargs="+",
        help="Online pool seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        help="Test seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--interceptor-port",
        type=int,
        default=8765,
        help="Interceptor port (default: 8765)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MIPRO Local Test: Iris")
    print("=" * 80)
    print(f"Task app URL: {args.task_app_url}")
    print(f"Rollout budget: {args.rollout_budget}")
    print(f"Interceptor port: {args.interceptor_port}")
    print("=" * 80)
    print()
    
    # Create config
    config = create_iris_mipro_config(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        bootstrap_seeds=args.bootstrap_seeds,
        online_seeds=args.online_seeds,
        test_seeds=args.test_seeds,
    )
    
    print("Configuration:")
    print(f"  Bootstrap seeds: {config.seeds.bootstrap}")
    print(f"  Online pool: {config.seeds.online}")
    print(f"  Test pool: {config.seeds.test}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Evaluations per iteration: {config.num_evaluations_per_iteration}")
    print()
    
    # Create LocalRuntime
    runtime = LocalRuntime(
        interceptor_host="localhost",
        interceptor_port=args.interceptor_port,
        task_app_host="localhost",
        task_app_port=8115,  # Iris port
    )
    
    # Create optimizer
    optimizer = MIPROOptimizer(
        job_id="iris_mipro_local",
        config=config,
        emitter=NoOpEmitter(),
        runtime=runtime,
        initial_prompt_config={
            "id": "iris_pattern",
            "name": "Iris Classification Pattern",
        },
    )
    
    print("Starting MIPRO optimization...")
    print()
    
    try:
        result = await optimizer.optimize()
        
        best_candidate = result.best_candidate
        best_prompt = optimizer.build_prompt_template(best_candidate)
        best_score = result.best_full_score or result.best_minibatch_score
        
        print()
        print("=" * 80)
        print("✅ MIPRO Optimization Complete!")
        print("=" * 80)
        print(f"Best Score: {best_score:.3f} ({best_score*100:.1f}%)")
        print(f"Best Prompt ID: {best_prompt.id}")
        print()
        
        print("Best Prompt Sections:")
        for i, section in enumerate(best_prompt.sections):
            role_str = section.role.value if hasattr(section.role, 'value') else str(section.role)
            print(f"  {i+1}. [{role_str}] {section.name}:")
            content_preview = section.content[:150] + "..." if len(section.content) > 150 else section.content
            print(f"     {content_preview}")
        
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ MIPRO Optimization Failed!")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Optimization interrupted by user")
        sys.exit(1)

