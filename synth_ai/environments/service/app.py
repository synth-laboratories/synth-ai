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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register built-in environments at import time
import synth_ai.environments.examples.sokoban.environment as sok

register_environment("Sokoban", sok.SokobanEnvironment)
import synth_ai.environments.examples.crafter_classic.environment as cc

register_environment("CrafterClassic", cc.CrafterClassicEnvironment)
import synth_ai.environments.examples.verilog.environment as ve

register_environment("Verilog", ve.VerilogEnvironment)
import synth_ai.environments.examples.tictactoe.environment as ttt

register_environment("TicTacToe", ttt.TicTacToeEnvironment)
import synth_ai.environments.examples.nethack.environment as nh

register_environment("NetHack", nh.NetHackEnvironment)
# AlgoTune excluded from package due to size/complexity
# import synth_ai.environments.examples.algotune.environment as at
# register_environment("AlgoTune", at.AlgoTuneEnvironment)
import synth_ai.environments.examples.minigrid.environment as mg

register_environment("MiniGrid", mg.MiniGridEnvironment)
import synth_ai.environments.examples.enron.environment as enron

register_environment("Enron", enron.EnronEnvironment)

app = FastAPI(title="Environment Service")


@app.on_event("startup")
async def startup_event():
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
    logger.info(f"Registered environments: {list_supported_env_types()}")


# Mount the main API router
app.include_router(api_router, tags=["environments"])
