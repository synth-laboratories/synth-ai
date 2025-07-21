"""CrafterClassicEnvironment — thin wrapper exposing CrafterEngine via StatefulEnvironment API."""

from __future__ import annotations

from typing import List, Optional, Any, Dict, Union
import dataclasses
import logging

# Import logging configuration to suppress JAX debug messages
from .config_logging import safe_compare

logger = logging.getLogger(__name__)

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


# --- Tool Definition ---
class CrafterActionInput(BaseModel):
    action: int = Field(..., description="Integer action for the Crafter environment.")


class CrafterInteractTool(AbstractTool):
    name = "interact"
    description = "Performs an action in the Crafter environment."
    call_schema = CrafterActionInput
    result_schema = ToolResult

    def __init__(self, engine: CrafterEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            action_to_pass = self.engine._validate_action_engine(validated_args.action)
            priv_state, pub_state = await self.engine._step_engine(action_to_pass)
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
    async def get_observation(
        self, pub: CrafterPublicState, priv: CrafterPrivateState
    ) -> InternalObservation:
        # Example: return a dictionary combining public and selected private info
        # Actual observation structure depends on agent's needs.
        obs_dict: Dict[str, Any] = dataclasses.asdict(pub)  # type: ignore
        obs_dict["reward_last_step"] = priv.reward_last_step
        obs_dict["total_reward_episode"] = priv.total_reward_episode
        obs_dict["terminated"] = priv.terminated
        obs_dict["truncated"] = priv.truncated
        if pub.error_info:
            obs_dict["tool_error"] = pub.error_info
        return obs_dict


class CrafterClassicEnvironment(StatefulEnvironment, ReproducibleEnvironment[CrafterEngine]):
    """Environment wrapper bridging agent tool‑calls to `crafter.Env` dynamics."""

    def __init__(
        self,
        task_instance: "CrafterTaskInstance",
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "CrafterClassic"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or SynthCrafterObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthCrafterObservationCallable()
        )
        self.engine = CrafterEngine(task_instance)

        self._interact_tool = CrafterInteractTool(self.engine)
        if self._interact_tool.name not in TOOL_REGISTRY:
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
        active_obs_cb = obs_cb or SynthCrafterObservationCallable()
        observation = await active_obs_cb.get_observation(pub, priv)
        if extra_obs and isinstance(observation, dict):
            observation.update(extra_obs)
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
        return env
