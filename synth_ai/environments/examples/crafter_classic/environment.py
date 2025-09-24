"""CrafterClassicEnvironment — thin wrapper exposing CrafterEngine via StatefulEnvironment API."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional, Union

# Import tracing abstractions
from synth_ai.tracing_v3.abstractions import (
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    TimeRecord,
)

# Import logging configuration to suppress JAX debug messages
from .config_logging import safe_compare

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.environment.tools import (
    TOOL_REGISTRY,
    AbstractTool,
    EnvToolCall,
    ToolResult,
    register_tool,
)
from synth_ai.environments.examples.crafter_classic.engine import (
    CrafterEngine,
    CrafterEngineSnapshot,
    CrafterPrivateState,
    CrafterPublicState,
)
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment


# --- Tool Definition ---
class CrafterActionInput(BaseModel):
    action: int = Field(..., description="Integer action for the Crafter environment.")


class CrafterInteractTool(AbstractTool):
    name = "interact"
    description = "Performs an action in the Crafter environment."
    call_schema = CrafterActionInput
    result_schema = ToolResult

    def __init__(self, engine: CrafterEngine, session_tracer: Optional[Any] = None):
        self.engine = engine
        self.session_tracer = session_tracer

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Store state before execution
            state_before = {"action_args": call.args}

            validated_args = self.call_schema(**call.args)
            action_to_pass = self.engine._validate_action_engine(validated_args.action)

            # Execute the engine step
            priv_state, pub_state = await self.engine._step_engine(action_to_pass)

            # Store state after execution
            state_after = {
                "engine_result": {"private_state": priv_state, "public_state": pub_state}
            }

            # Record runtime event for tool execution
            if (
                self.session_tracer
                and hasattr(self.session_tracer, "current_session")
                and self.session_tracer.current_session
            ):
                runtime_execution_event = RuntimeEvent()
                runtime_execution_event.time_record = TimeRecord()
                runtime_execution_event.time_record.event_time = time.time()
                runtime_execution_event.time_record.message_time = None
                runtime_execution_event.system_instance_id = "crafter_interact_tool"
                runtime_execution_event.system_state_before = state_before
                runtime_execution_event.system_state_after = state_after
                runtime_execution_event.actions = [action_to_pass]
                runtime_execution_event.metadata = {"execution_step": "engine_action"}
                # Add directly to event history, bypassing timestep requirement
                self.session_tracer.current_session.add_event(runtime_execution_event)

            return ToolResult(
                ok=True,
                payload={
                    "public_state": pub_state,
                    "private_state": priv_state,
                },
            )
        except Exception as e:
            pub_state_on_error = self.engine._get_public_state_from_env()  # Use engine helper
            # Get a safe private state for error cases
            health_dead = safe_compare(0, self.engine.env._player.health, ">=")
            step_exceeded = safe_compare(self.engine.env._length, self.engine.env._step, "<=")
            priv_state_on_error = self.engine._get_private_state_from_env(
                0, health_dead, step_exceeded
            )
            return ToolResult(
                ok=False,
                error=str(e),
                payload={
                    "public_state": pub_state_on_error,
                    "private_state": priv_state_on_error,
                },
            )


# Default observation callable (can be customized via __init__)
class SynthCrafterObservationCallable(GetObservationCallable):
    """Default observation: public state dict + per-step reward/flags.

    Additionally computes a small local semantic patch centered on the player
    to simplify visualization on the client. The patch is exposed under the
    key `semantic_map_patch7` as a list-of-lists of ints (7x7 unless the
    semantic map is smaller, in which case it is cropped at edges).
    """

    def __init__(self, view_size: int = 7) -> None:
        self.view_size = max(1, int(view_size))

    async def get_observation(
        self, pub: CrafterPublicState, priv: CrafterPrivateState
    ) -> InternalObservation:
        obs_dict: Dict[str, Any] = dataclasses.asdict(pub)  # type: ignore
        obs_dict["reward_last_step"] = priv.reward_last_step
        obs_dict["total_reward_episode"] = priv.total_reward_episode
        obs_dict["terminated"] = priv.terminated
        obs_dict["truncated"] = priv.truncated
        if pub.error_info:
            obs_dict["tool_error"] = pub.error_info

        # Derive a simple local semantic patch around the player for easy rendering
        try:
            sem = pub.semantic_map
            if sem is not None:
                rows = int(getattr(sem, "shape", [0, 0])[0])  # type: ignore
                cols = int(getattr(sem, "shape", [0, 0])[1])  # type: ignore
                if rows > 0 and cols > 0:
                    px, py = int(pub.player_position[0]), int(pub.player_position[1])
                    half = max(1, self.view_size // 2)
                    x0, y0 = px - half, py - half
                    x1, y1 = px + half, py + half
                    patch: list[list[int]] = []
                    for gy in range(y0, y1 + 1):
                        row_vals: list[int] = []
                        for gx in range(x0, x1 + 1):
                            if 0 <= gy < rows and 0 <= gx < cols:
                                try:
                                    val = int(sem[gy, gx])  # type: ignore[index]
                                except Exception:
                                    val = 0
                            else:
                                val = 0
                            row_vals.append(val)
                        patch.append(row_vals)
                    obs_dict["semantic_map_patch7"] = patch
        except Exception:
            # Best-effort; omit patch on error
            pass

        return obs_dict


class CrafterClassicEnvironment(StatefulEnvironment, ReproducibleEnvironment[CrafterEngine]):
    """Environment wrapper bridging agent tool‑calls to `crafter.Env` dynamics."""

    def __init__(
        self,
        task_instance: "CrafterTaskInstance",
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
        session_tracer: Optional[Any] = None,  # SessionTracer from higher level
    ) -> None:
        self.name = "CrafterClassic"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or SynthCrafterObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthCrafterObservationCallable()
        )
        self.engine = CrafterEngine(task_instance)
        self.session_tracer = session_tracer  # Store tracer for runtime events

        self._interact_tool = CrafterInteractTool(self.engine, session_tracer=session_tracer)
        if self._interact_tool.name not in TOOL_REGISTRY:
            register_tool(self._interact_tool)

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle helpers
    # ────────────────────────────────────────────────────────────────────

    async def initialize(self, seed: Optional[int] = None) -> InternalObservation:  # type: ignore[override]
        # Check if seed was provided in task instance metadata
        if (
            seed is None
            and hasattr(self.task_instance, "metadata")
            and hasattr(self.task_instance.metadata, "seed")
        ):
            seed = self.task_instance.metadata.seed
        # Check if seed was provided in initial_engine_snapshot
        elif (
            seed is None
            and hasattr(self.task_instance, "initial_engine_snapshot")
            and isinstance(self.task_instance.initial_engine_snapshot, dict)
        ):
            seed = self.task_instance.initial_engine_snapshot.get("seed")

        # Initialize with seed from various sources

        priv, pub = await self.engine._reset_engine(seed=seed)
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

        # Normalize and validate to a single EnvToolCall (same as Sokoban)
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
        if agent_call.tool != "interact":
            raise ValueError(f"Unknown tool: {agent_call.tool}. Expected 'interact'.")

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
            runtime_validation_event.system_instance_id = "crafter_environment"
            runtime_validation_event.system_state_before = state_before
            runtime_validation_event.system_state_after = {"validated_call": agent_call}
            runtime_validation_event.metadata = {"validation_step": "tool_call_validation"}
            # Add directly to event history, bypassing timestep requirement
            self.session_tracer.current_session.add_event(runtime_validation_event)

        return agent_call

    async def step(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> InternalObservation:  # type: ignore[override]
        step_start_time = time.time()
        agent_call = self.validate_tool_calls(tool_calls)
        interact_start = time.time()
        tool_result: ToolResult = await self._interact_tool(agent_call)
        interact_time = time.time() - interact_start

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

        obs = await self._to_observation(
            priv_state, pub_state, self.custom_step_observation_callable
        )
        total_step_time = time.time() - step_start_time
        logger.info(
            f"CrafterClassic step completed in {total_step_time:.3f}s (interact: {interact_time:.3f}s)"
        )
        return obs

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
    ) -> "CrafterClassicEnvironment":
        eng = await CrafterEngine._deserialize_engine(snapshot, task_instance)
        env = cls(task_instance)
        env.engine = eng
        # CRITICAL: Update the interact tool to use the new engine!
        env._interact_tool.engine = eng
        return env
