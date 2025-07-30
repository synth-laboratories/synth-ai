"""MiniGrid environment example for synth_env.

This module provides a comprehensive implementation of MiniGrid environments
with full state management, tool-based interaction, and task generation.
"""

from synth_ai.environments.examples.minigrid.engine import (
    MiniGridEngine,
    MiniGridPublicState,
    MiniGridPrivateState,
    MiniGridGoalReachedComponent,
    MiniGridStepPenaltyComponent,
    MiniGridObservationCallable,
    MiniGridCheckpointObservationCallable,
)
from synth_ai.environments.examples.minigrid.environment import (
    MiniGridEnvironment,
    MiniGridInteractTool,
    MiniGridActionInput,
)
from synth_ai.environments.examples.minigrid.taskset import (
    MiniGridTaskInstance,
    MiniGridTaskInstanceMetadata,
    DEFAULT_MINIGRID_TASK,
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
