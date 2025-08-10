"""CrafterCustomEnvironment — Custom Crafter with configurable world generation."""

from __future__ import annotations

from typing import List, Optional, Any, Dict, Union
import dataclasses
import logging
import time

# Import logging configuration to suppress JAX debug messages
from synth_ai.environments.examples.crafter_classic.config_logging import safe_compare

# Import tracing abstractions
from synth_ai.tracing_v3.abstractions import (
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    TimeRecord,
)

logger = logging.getLogger(__name__)

# Import the base Crafter components
from synth_ai.environments.examples.crafter_classic.engine import (
    CrafterEngine,
    CrafterPrivateState,
    CrafterPublicState,
    CrafterEngineSnapshot,
)
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
    TOOL_REGISTRY,
    register_tool,
)
from pydantic import BaseModel, Field


# Use the same tool and observation classes as CrafterClassic
from synth_ai.environments.examples.crafter_classic.environment import (
    CrafterActionInput,
    CrafterInteractTool,
    SynthCrafterObservationCallable,
)


class CrafterCustomEnvironment(StatefulEnvironment, ReproducibleEnvironment[CrafterEngine]):
    """Custom Crafter environment with configurable world generation."""

    def __init__(
        self,
        task_instance: "CrafterTaskInstance",
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
        session_tracer: Optional[Any] = None,  # SessionTracer from higher level
    ) -> None:
        self.name = "CrafterCustom"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or SynthCrafterObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthCrafterObservationCallable()
        )

        # Ensure task instance has world configuration
        if hasattr(task_instance, "metadata"):
            logger.info(
                f"Creating CrafterCustom with world_config: {getattr(task_instance.metadata, 'world_config', 'default')}"
            )

        self.engine = CrafterEngine(task_instance)
        self.session_tracer = session_tracer  # Store tracer for runtime events

        self._interact_tool = CrafterInteractTool(self.engine, session_tracer=session_tracer)

        # Register tool with a unique name for this environment
        tool_name = f"{self.name.lower()}_interact"
        if tool_name not in TOOL_REGISTRY:
            # Create a copy of the tool with the custom name
            self._interact_tool.name = tool_name
            register_tool(self._interact_tool)

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle helpers
    # ────────────────────────────────────────────────────────────────────

    async def initialize(self) -> InternalObservation:  # type: ignore[override]
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:  # type: ignore[override]
        pub = self.engine._get_public_state_from_env()
        priv = self.engine._get_private_state_from_env(0, True, False)  # Terminated state
        priv.terminated = True
        obs_dict = {"status": "Environment terminated."}
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable, extra_obs=obs_dict
        )

    # ────────────────────────────────────────────────────────────────────
    # Step + checkpoint
    # ────────────────────────────────────────────────────────────────────

    def validate_tool_calls(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> EnvToolCall:
        # Store the original tool calls for tracing
        state_before = {"tool_calls": tool_calls}

        # Normalize and validate to a single EnvToolCall
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Received empty list of tool calls.")
            if isinstance(tool_calls[0], list):
                if not tool_calls[0]:
                    raise ValueError("Received empty inner list of tool calls.")
                agent_call = tool_calls[0][0]
            else:
                agent_call = tool_calls[0]
        elif isinstance(tool_calls, EnvToolCall):
            agent_call = tool_calls
        else:
            raise TypeError(f"Unexpected type for tool_calls: {type(tool_calls)}")

        if not isinstance(agent_call, EnvToolCall):
            raise TypeError(f"Processed call is not EnvToolCall: {type(agent_call)}")

        # Accept both "interact" and "craftercustom_interact"
        if agent_call.tool not in ["interact", f"{self.name.lower()}_interact"]:
            raise ValueError(
                f"Unknown tool: {agent_call.tool}. Expected 'interact' or '{self.name.lower()}_interact'."
            )

        # Record runtime event for tool call validation
        if (
            self.session_tracer
            and hasattr(self.session_tracer, "current_session")
            and self.session_tracer.current_session
        ):
            runtime_validation_event = RuntimeEvent()
            runtime_validation_event.time_record = TimeRecord()
            runtime_validation_event.time_record.event_time = time.time()
            runtime_validation_event.time_record.message_time = None
            runtime_validation_event.system_instance_id = "crafter_custom_environment"
            runtime_validation_event.system_state_before = state_before
            runtime_validation_event.system_state_after = {"validated_call": agent_call}
            runtime_validation_event.metadata = {"validation_step": "tool_call_validation"}
            # Add directly to event history, bypassing timestep requirement
            self.session_tracer.current_session.add_event(runtime_validation_event)

        return agent_call

    async def step(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> InternalObservation:  # type: ignore[override]
        agent_call = self.validate_tool_calls(tool_calls)
        tool_result: ToolResult = await self._interact_tool(agent_call)

        payload_dict = tool_result.payload
        pub_state: CrafterPublicState
        priv_state: CrafterPrivateState

        if tool_result.ok:
            # payload contains the actual state objects from the interact tool
            priv_state = payload_dict.get("private_state")
            pub_state = payload_dict.get("public_state")

            # Validate we got the expected state objects
            if not isinstance(priv_state, CrafterPrivateState) or not isinstance(
                pub_state, CrafterPublicState
            ):
                logger.error(
                    f"Invalid state types in payload: priv={type(priv_state)}, pub={type(pub_state)}"
                )
                # Fall back to getting current state
                pub_state = self.engine._get_public_state_from_env()
                health_dead = safe_compare(0, self.engine.env._player.health, ">=")
                step_exceeded = safe_compare(self.engine.env._length, self.engine.env._step, "<=")
                priv_state = self.engine._get_private_state_from_env(0, health_dead, step_exceeded)
                pub_state.error_info = "Invalid state types in tool result"
        else:
            # Tool call failed, use states from payload if available, otherwise get current state
            priv_state = payload_dict.get("private_state")
            pub_state = payload_dict.get("public_state")

            if not isinstance(priv_state, CrafterPrivateState) or not isinstance(
                pub_state, CrafterPublicState
            ):
                # Fall back to getting current state
                pub_state = self.engine._get_public_state_from_env()
                health_dead = safe_compare(0, self.engine.env._player.health, ">=")
                step_exceeded = safe_compare(self.engine.env._length, self.engine.env._step, "<=")
                priv_state = self.engine._get_private_state_from_env(0, health_dead, step_exceeded)

            if tool_result.error:
                pub_state.error_info = tool_result.error

        return await self._to_observation(
            priv_state, pub_state, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:  # type: ignore[override]
        engine_snapshot: CrafterEngineSnapshot = await self.engine._serialize_engine()
        priv = self.engine._get_private_state_from_env(0, False, False)  # Get current state for obs
        pub = self.engine._get_public_state_from_env()
        obs_data = await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )
        if isinstance(obs_data, dict):
            obs_data["engine_snapshot_data"] = engine_snapshot.model_dump()
        return obs_data

    async def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current environment configuration."""
        metadata = {
            "environment_type": "CrafterCustom",
            "engine_seed": getattr(self.engine.env, "_seed", None),
            "world_area": self.engine.env._area,
            "max_steps": self.engine.env._length,
            "current_step": self.engine.env._step,
        }

        # Add task instance metadata
        if hasattr(self.task_instance, "metadata"):
            task_metadata = self.task_instance.metadata
            metadata.update(
                {
                    "difficulty": getattr(task_metadata, "difficulty", None),
                    "world_config": getattr(task_metadata, "world_config", None),
                    "world_config_path": getattr(task_metadata, "world_config_path", None),
                    "num_trees_radius": getattr(task_metadata, "num_trees_radius", None),
                    "num_cows_radius": getattr(task_metadata, "num_cows_radius", None),
                    "num_hostiles_radius": getattr(task_metadata, "num_hostiles_radius", None),
                }
            )

        # Add current world statistics
        if hasattr(self.engine, "env") and hasattr(self.engine.env, "_world"):
            world = self.engine.env._world
            object_counts = {}

            for obj in world._objects:
                if obj is None:
                    continue
                obj_type = type(obj).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

            metadata["world_object_counts"] = object_counts

        return metadata

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    async def _to_observation(
        self,
        priv: CrafterPrivateState,
        pub: CrafterPublicState,
        obs_cb: Optional[GetObservationCallable],
        extra_obs: Optional[Dict[str, Any]] = None,
    ) -> InternalObservation:
        # Store state before observation generation
        state_before = {"private_state": priv, "public_state": pub}

        active_obs_cb = obs_cb or SynthCrafterObservationCallable()
        observation = await active_obs_cb.get_observation(pub, priv)
        if extra_obs and isinstance(observation, dict):
            observation.update(extra_obs)

        # Record runtime event for observation generation
        if (
            self.session_tracer
            and hasattr(self.session_tracer, "current_session")
            and self.session_tracer.current_session
        ):
            runtime_obs_event = RuntimeEvent()
            runtime_obs_event.time_record = TimeRecord()
            runtime_obs_event.time_record.event_time = time.time()
            runtime_obs_event.time_record.message_time = None
            runtime_obs_event.system_instance_id = "observation_generator"
            runtime_obs_event.system_state_before = state_before
            runtime_obs_event.system_state_after = {"observation": observation}
            runtime_obs_event.metadata = {"observation_step": "state_to_obs_conversion"}
            # Add directly to event history, bypassing timestep requirement
            self.session_tracer.current_session.add_event(runtime_obs_event)

        return observation

    # ────────────────────────────────────────────────────────────────────
    # ReproducibleEnvironment plumbing
    # ────────────────────────────────────────────────────────────────────

    async def _serialize_engine(self) -> CrafterEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: CrafterEngineSnapshot, task_instance: "CrafterTaskInstance"
    ) -> "CrafterCustomEnvironment":
        eng = await CrafterEngine._deserialize_engine(snapshot, task_instance)
        env = cls(task_instance)
        env.engine = eng
        # CRITICAL: Update the interact tool to use the new engine!
        env._interact_tool.engine = eng
        return env
