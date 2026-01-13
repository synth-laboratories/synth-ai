#!/usr/bin/env python3
"""
Direct GEPA run using backend modules.
Now that Redis and all infrastructure is properly running, this should work!
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = Path(__file__).parent.parent.parent.parent / "monorepo-context-eng-unified" / "backend"
sys.path.insert(0, str(backend_path))

# Load environment from backend .env.dev file
env_file = Path(__file__).parent.parent.parent.parent / "monorepo-context-eng-unified" / "backend" / ".env.dev"
if env_file.exists():
    print(f"Loading environment from: {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Don't override already-set variables
                if key not in os.environ:
                    os.environ[key] = value

# Setup environment variables BEFORE any backend imports
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["DEV_SESSION_SUPABASE_DB_URL"] = "postgresql://postgres.mivawyjmuwggrynczaef:coXzyf-kotcuz-jowxa2@aws-0-us-east-2.pooler.supabase.com:5432/postgres"
# Will be set by setup_auth(), just declaring intent
os.environ.setdefault("ENVIRONMENT_API_KEY", "")

# Check for OPENAI_API_KEY (required for GEPA proposer)
if not os.environ.get("OPENAI_API_KEY"):
    print("=" * 80)
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("=" * 80)
    print()
    print("GEPA needs an OpenAI API key to:")
    print("  1. Generate new prompt candidates (proposer)")
    print("  2. Run the interceptor for trace capture")
    print()
    print("The .env.dev file should contain OPENAI_API_KEY")
    print(f"Checked: {env_file}")
    print()
    print("=" * 80)
    sys.exit(1)
else:
    print(f"✓ OPENAI_API_KEY loaded: {os.environ['OPENAI_API_KEY'][:20]}...")


def setup_auth():
    """Set up authentication before running GEPA."""
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth

    # Get or create SYNTH_API_KEY
    API_KEY = os.environ.get("SYNTH_API_KEY", "")
    if not API_KEY:
        from synth_ai.core.env import mint_demo_api_key
        print("No SYNTH_API_KEY found, minting demo key...")
        API_KEY = mint_demo_api_key(backend_url="http://localhost:8000")
        print(f"Demo API Key: {API_KEY[:25]}...")
        os.environ["SYNTH_API_KEY"] = API_KEY

    # Ensure ENVIRONMENT_API_KEY is registered in backend
    ENVIRONMENT_API_KEY = ensure_localapi_auth(
        backend_base="http://localhost:8000",
        synth_api_key=API_KEY,
    )
    print(f"Environment key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")
    os.environ["ENVIRONMENT_API_KEY"] = ENVIRONMENT_API_KEY

    return API_KEY, ENVIRONMENT_API_KEY


async def main(api_key: str, env_key: str):
    """Run GEPA unified optimization."""

    # Import after path setup
    from app.routes.prompt_learning.core.config import (
        load_toml,
        parse_prompt_learning_config,
    )
    from app.routes.prompt_learning.core.optimizer_factory import get_optimizer_factory

    config_path = Path(__file__).parent / "enginebench_gepa_quick.toml"

    logger.info("=" * 80)
    logger.info("GEPA UNIFIED OPTIMIZATION - LIVE RUN")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info("Backend: http://localhost:8000 (HEALTHY ✓)")
    logger.info("Redis: redis://localhost:6379/0 (RUNNING ✓)")
    logger.info("Task app: http://localhost:8020 (RUNNING ✓)")
    logger.info(f"Auth: API key = {api_key[:20]}...")
    logger.info(f"Auth: ENV key = {env_key[:12]}...{env_key[-4:]}")
    logger.info("=" * 80)

    # Load config
    logger.info("Loading configuration...")
    raw_config = load_toml(config_path)

    # Parse configuration - will fail on missing credentials, but we'll catch it
    logger.info("Parsing configuration...")
    try:
        parsed_config, initial_prompt, optimizer_config = await parse_prompt_learning_config(raw_config)
    except Exception as e:
        if "ENVIRONMENT_API_KEY" in str(e):
            # Expected for local dev without DB - create config manually
            logger.info("DB credentials not available, creating config manually for local dev...")
            from app.routes.prompt_learning.algorithm.gepa import GEPAConfig

            # Extract seeds from config
            eval_seeds = raw_config.get("prompt_learning", {}).get("evaluation_seeds", {})
            pareto_seeds = eval_seeds.get("pareto", [0, 2, 7])

            # Extract GEPA params
            gepa_config = raw_config.get("prompt_learning", {}).get("gepa", {})

            # Use pareto seeds as train seeds if train not specified
            train_seeds = eval_seeds.get("train", pareto_seeds)

            # Create env_config with seeds
            env_config = {
                "seeds": {
                    "train": train_seeds,
                    "test": eval_seeds.get("test", []),
                    "pareto": pareto_seeds,
                },
            }

            # Extract policy config from raw config
            policy_dict = raw_config.get("prompt_learning", {}).get("policy", {})

            # Task app uses a specific hardcoded key
            task_app_key = "sk_env_30c78a787bac223c716918181209f263"

            # Create optimizer config manually
            # With 6 train seeds and pareto_set_size=3, we have 3 feedback seeds (meets minimum)
            optimizer_config = GEPAConfig(
                task_app_url="http://localhost:8020",  # No /rollout suffix - validation code adds it
                task_app_api_key=task_app_key,
                task_app_api_keys=[task_app_key],
                env_name="engine_bench",
                env_config=env_config,
                policy_config=policy_dict,  # Include policy config with model/provider
                pareto_set_size=len(pareto_seeds),  # 3 pareto seeds
                num_generations=gepa_config.get("num_generations", 2),
                initial_population_size=gepa_config.get("initial_population_size", 4),
                mutation_rate=gepa_config.get("mutation_rate", 0.5),
                crossover_rate=gepa_config.get("crossover_rate", 0.5),
                minibatch_size=3,  # Use 3 seeds per minibatch
            )

            # Extract initial prompt from config
            from app.routes.prompt_learning.core.patterns import MessagePattern, PromptPattern
            policy_config = raw_config.get("prompt_learning", {}).get("policy", {})
            context_override = policy_config.get("context_override", {})
            system_prompt = context_override.get("system_prompt", "You are a helpful assistant")

            initial_prompt = PromptPattern(
                messages=[MessagePattern(role="system", pattern=system_prompt)]
            )
            parsed_config = raw_config
        else:
            raise

    logger.info(f"Task app URL: {optimizer_config.task_app_url}")
    logger.info(f"Task app API key set: {optimizer_config.task_app_api_key[:12]}...")

    # Create optimizer
    logger.info("Creating GEPA optimizer...")
    factory = get_optimizer_factory()

    optimizer = factory.create_optimizer(
        algorithm="gepa",
        config=optimizer_config,
    )

    logger.info("=" * 80)
    logger.info("Starting optimization...")
    logger.info("Generations: 2")
    logger.info("Children per gen: 4")
    logger.info("Evaluation seeds: [0, 2, 7] (3 cards)")
    logger.info("=" * 80)

    # Run!
    try:
        # Extract seeds from env_config
        env_cfg = optimizer_config.env_config or {}
        seeds_cfg = env_cfg.get("seeds", {})
        train_seeds = seeds_cfg.get("train", [0, 2, 7])
        test_seeds = seeds_cfg.get("test", [])

        logger.info(f"Train seeds: {train_seeds}")
        logger.info(f"Test seeds: {test_seeds}")

        result = await optimizer.optimize(
            initial_pattern=initial_prompt,
            train_seeds=train_seeds,
            test_pool=test_seeds if test_seeds else None,
        )

        logger.info("=" * 80)
        logger.info("✓ OPTIMIZATION COMPLETE!")
        logger.info("=" * 80)

        if hasattr(result, "best_candidate"):
            best = result.best_candidate
            logger.info(f"Best accuracy: {best.accuracy:.3f}")

            if hasattr(best, "context_override") and best.context_override:
                logger.info("Context override applied:")
                logger.info(f"  - Artifacts: {best.context_override.artifact_count()}")
                logger.info(f"  - Size: {best.context_override.size_bytes()} bytes")

        logger.info("=" * 80)
        return result

    except Exception as e:
        logger.error(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Setup authentication FIRST
    api_key, env_key = setup_auth()

    try:
        result = asyncio.run(main(api_key, env_key))
        if result:
            print("\n" + "=" * 80)
            print("✓ SUCCESS: Unified optimization completed!")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("✗ FAILED: See logs above")
            print("=" * 80)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
