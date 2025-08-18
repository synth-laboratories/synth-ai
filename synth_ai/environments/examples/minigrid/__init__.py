"""MiniGrid environment example for synth_env.

This module provides a comprehensive implementation of MiniGrid environments
with full state management, tool-based interaction, and task generation.
"""

from synth_ai.environments.examples.minigrid.engine import (
    MiniGridCheckpointObservationCallable,
    MiniGridEngine,
    MiniGridGoalReachedComponent,
    MiniGridObservationCallable,
    MiniGridPrivateState,
    MiniGridPublicState,
    MiniGridStepPenaltyComponent,
)
from synth_ai.environments.examples.minigrid.environment import (
    MiniGridActionInput,
    MiniGridEnvironment,
    MiniGridInteractTool,
)
from synth_ai.environments.examples.minigrid.taskset import (
    DEFAULT_MINIGRID_TASK,
    MiniGridTaskInstance,
    MiniGridTaskInstanceMetadata,
    create_minigrid_taskset,
    taskset,
)

__all__ = [
    # Engine
    "MiniGridEngine",
    "MiniGridPublicState",
    "MiniGridPrivateState",
    "MiniGridGoalReachedComponent",
    "MiniGridStepPenaltyComponent",
    "MiniGridObservationCallable",
    "MiniGridCheckpointObservationCallable",
    # Environment
    "MiniGridEnvironment",
    "MiniGridInteractTool",
    "MiniGridActionInput",
    # TaskSet
    "MiniGridTaskInstance",
    "MiniGridTaskInstanceMetadata",
    "DEFAULT_MINIGRID_TASK",
    "create_minigrid_taskset",
    "taskset",
]
