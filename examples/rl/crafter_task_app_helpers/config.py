from __future__ import annotations

from synth_ai.environments.examples.crafter_classic.engine import (
    CRAFTER_ACTION_MAP,
)

# Environment name for API endpoints
ENV_NAME = "crafter"

# Available actions for the crafter environment
ACTION_SPACE = list(CRAFTER_ACTION_MAP.keys())

