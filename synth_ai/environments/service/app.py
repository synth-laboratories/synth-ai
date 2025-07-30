import sys
import os  # Added to ensure os is available before use

# Ensure local 'src' directory is on PYTHONPATH for dev installs
# Current file: <repo>/src/synth_env/service/app.py
# We want to add <repo>/src to sys.path (two levels up)
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

print(f"SYS.PATH IN APP.PY: {sys.path}")
import logging

from fastapi import FastAPI
from synth_ai.environments.service.registry import list_supported_env_types, register_environment
from synth_ai.environments.service.core_routes import api_router
from synth_ai.environments.service.external_registry import (
    ExternalRegistryConfig,
    load_external_environments,
)

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
