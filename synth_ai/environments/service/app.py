import os  # Added to ensure os is available before use
import sys

# Ensure repository root is on PYTHONPATH for dev installs
# Current file path: <repo>/synth_ai/environments/service/app.py
# We want sys.path to include <repo>, NOT <repo>/synth_ai to avoid shadowing stdlib 'http'
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_repo_root = os.path.abspath(os.path.join(_pkg_dir, ".."))
# If the package directory was previously added, remove it to prevent top-level shadowing
try:
    while _pkg_dir in sys.path:
        sys.path.remove(_pkg_dir)
except Exception:
    pass
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

print(f"SYS.PATH IN APP.PY: {sys.path}")
import logging

from fastapi import FastAPI
from synth_ai.environments.service.core_routes import api_router
from synth_ai.environments.service.external_registry import (
    ExternalRegistryConfig,
    load_external_environments,
)
from synth_ai.environments.service.registry import list_supported_env_types, register_environment

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Also configure uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

# Register built-in environments at import time
import synth_ai.environments.examples.crafter_classic.environment as cc

register_environment("CrafterClassic", cc.CrafterClassicEnvironment)
import synth_ai.environments.examples.crafter_custom.environment as ccustom

register_environment("CrafterCustom", ccustom.CrafterCustomEnvironment)

# Register Wordle example environment
try:
    import synth_ai.environments.examples.wordle.environment as wordle_mod

    register_environment("Wordle", wordle_mod.WordleEnvironment)
except Exception as _e:
    # Keep service robust even if example env import fails
    logging.getLogger(__name__).warning(f"Wordle env not registered: {_e}")

# Register Bandit example environment
try:
    import synth_ai.environments.examples.bandit.environment as bandit_mod

    register_environment("Bandit", bandit_mod.BanditEnvironment)
except Exception as _e:
    logging.getLogger(__name__).warning(f"Bandit env not registered: {_e}")

app = FastAPI(title="Environment Service")


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Starting Environment Service")
    logger.info("=" * 60)
    """Load external environments on startup."""
    # Support configuration-based loading for external environments
    # You can set EXTERNAL_ENVIRONMENTS env var with JSON config
    external_config = os.getenv("EXTERNAL_ENVIRONMENTS")
    if external_config:
        try:
            import json

            config_data = json.loads(external_config)
            config = ExternalRegistryConfig(
                external_environments=config_data.get("external_environments", [])
            )
            load_external_environments(config)
        except Exception as e:
            logger.error(f"Failed to load external environment config: {e}")

    # Log all registered environments
    env_types = list_supported_env_types()
    logger.info(f"Registered environments: {env_types}")
    logger.info(f"Total environments available: {len(env_types)}")
    logger.info("Environment Service startup complete")
    logger.info("=" * 60)


# Mount the main API router
app.include_router(api_router, tags=["environments"])
