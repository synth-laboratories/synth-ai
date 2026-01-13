#!/usr/bin/env python3
"""
Run GEPA with unified context optimization for EngineBench.

This script demonstrates the full unified optimization pipeline:
1. UnifiedProposer generates context mutations
2. Task app receives and applies context overrides
3. GEPA evolves both prompts AND context together
4. Multi-objective selection balances accuracy vs complexity
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent.parent / "monorepo-context-eng-unified" / "backend"
sys.path.insert(0, str(backend_path))

from app.routes.prompt_learning.core.config import (
    load_toml,
    parse_prompt_learning_config,
)
from app.routes.prompt_learning.core.optimizer_factory import get_optimizer_factory
from app.routes.prompt_learning.core.runtime import LocalRuntime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_gepa_unified():
    """Run GEPA with unified context optimization."""

    # Load config
    config_path = Path(__file__).parent / "enginebench_gepa.toml"
    logger.info(f"Loading config from: {config_path}")

    raw_config = load_toml(config_path)

    # Parse config
    logger.info("Parsing config...")
    parsed_config, initial_prompt, optimizer_config = await parse_prompt_learning_config(
        raw_config
    )

    # Create local runtime
    runtime = LocalRuntime()

    # Adjust task app URL
    task_app_url = runtime.get_task_app_base_url(
        optimizer_config.task_app_url
    )
    logger.info(f"Task app URL: {task_app_url}")
    optimizer_config.task_app_url = task_app_url

    # Create optimizer factory
    factory = get_optimizer_factory()

    # Create GEPA optimizer
    logger.info("Creating GEPA optimizer...")
    job_id = f"enginebench_unified_{int(asyncio.get_event_loop().time())}"

    optimizer_kwargs = {}
    if hasattr(initial_prompt, "to_dict"):
        optimizer_kwargs["initial_template"] = initial_prompt.to_dict()
    else:
        optimizer_kwargs["initial_template"] = initial_prompt

    optimizer = factory.create_optimizer(
        algorithm="gepa",
        optimizer_config=optimizer_config,
        job_id=job_id,
        **optimizer_kwargs
    )

    # Run optimization
    logger.info("=" * 80)
    logger.info("Starting GEPA Unified Context Optimization")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info("Task: EngineBench Rust Code Generation")
    logger.info(f"Generations: {getattr(optimizer_config, 'num_generations', 5)}")
    logger.info(f"Population size: {getattr(optimizer_config, 'initial_population_size', 8)}")
    logger.info(f"Evaluation seeds: {len(optimizer_config.pareto_set_size)} cards")
    logger.info("=" * 80)

    # Check if unified optimization is enabled
    if hasattr(optimizer_config, 'optimize_context_artifacts'):
        logger.info("✓ UNIFIED OPTIMIZATION ENABLED")
        logger.info("  - System prompt optimization: ON")
        logger.info("  - Context artifact optimization: ON")
        logger.info("  - Multi-objective selection: ON (accuracy + complexity)")
    else:
        logger.info("✗ Running baseline GEPA (prompt-only)")

    logger.info("=" * 80)

    try:
        # Run optimization
        result = await optimizer.optimize(
            initial_template=optimizer_kwargs.get("initial_template"),
        )

        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)

        # Display results
        if hasattr(result, "best_candidate"):
            best = result.best_candidate
            logger.info(f"Best accuracy: {best.accuracy:.3f}")

            # Check if context override was used
            if hasattr(best, "context_override") and best.context_override:
                logger.info("✓ Context override applied:")
                logger.info(f"  - Artifacts: {best.context_override.artifact_count()}")
                logger.info(f"  - Total size: {best.context_override.size_bytes()} bytes")

                if best.context_override.file_artifacts:
                    logger.info(f"  - Files: {list(best.context_override.file_artifacts.keys())}")
                if best.context_override.preflight_script:
                    logger.info(f"  - Preflight script: {len(best.context_override.preflight_script)} chars")
                if best.context_override.env_vars:
                    logger.info(f"  - Env vars: {list(best.context_override.env_vars.keys())}")

        logger.info("=" * 80)
        logger.info(f"Results saved to: {runtime.get_results_dir() / job_id}")
        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main entry point."""

    # Check environment
    if not os.getenv("REDIS_URL"):
        logger.warning("REDIS_URL not set - using default: redis://localhost:6379/0")
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    # Run optimization
    try:
        result = asyncio.run(run_gepa_unified())
        logger.info("✓ Optimization completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
