"""NetHack environment wrapper for synth-env framework."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.environment.tools import AbstractTool, EnvToolCall, ToolResult
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.tasks.core import TaskInstance

from .engine import (
    NetHackCheckpointObservationCallable,
    NetHackEngine,
    NetHackObservationCallable,
    NetHackPrivateState,
    NetHackPublicState,
)
from .helpers import (
    MENU_ACTIONS,
    NETHACK_ACTIONS,
    get_action_description,
    validate_action,
)


class NetHackActionInput(BaseModel):
    """Pydantic model for NetHack action validation."""

    action: str  # Action string from NETHACK_ACTIONS or MENU_ACTIONS


class NetHackInteractTool(AbstractTool):
    """Tool for performing actions in NetHack."""

    name = "interact"
    description = (
        "Perform an action in the NetHack dungeon. Available actions include "
        "movement (north, south, east, west), combat (fight), inventory management "
        "(inventory, pickup, drop), and many others. In menus, use letter keys (a-z) "
        "or numbers (0-9) to select options, or 'escape' to cancel."
    )
    call_schema = NetHackActionInput
    result_schema = ToolResult

    def __init__(self, engine: NetHackEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        """Execute the interact tool."""
        try:
            action = call.args["action"]  # Will KeyError if missing

            # Get current game state for validation
            priv, pub = self.engine.get_current_states_for_observation()
            game_state = {
                "in_menu": pub.in_menu,
                "terminated": pub.terminated,
                "stairs_here": False,  # Would be determined from map parsing
            }

            # Validate action
            is_valid, error_msg = validate_action(action, game_state)
            if not is_valid:
                return ToolResult(
                    ok=False,
                    error=error_msg or f"Invalid action: {action}",
                    payload={"public_state": pub, "private_state": priv},
                )

            # Execute action
            private_state, public_state = await self.engine._step_engine(action)

            return ToolResult(
                ok=True,
                payload={"public_state": public_state, "private_state": private_state},
            )

        except Exception as e:
            # Return current state even on error
            priv, pub = self.engine.get_current_states_for_observation()
            return ToolResult(
                ok=False,
                error=str(e),
                payload={"public_state": pub, "private_state": priv},
            )


class NetHackEnvironment(StatefulEnvironment, ReproducibleEnvironment[NetHackEngine]):
    """NetHack environment implementation."""

    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        """Initialize NetHack environment."""
        self.name = "NetHack"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or NetHackObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or NetHackCheckpointObservationCallable()
        )
        self.engine = NetHackEngine(task_instance)
        self._interact_tool = NetHackInteractTool(self.engine)

    async def initialize(self) -> InternalObservation:
        """Initialize the environment and return initial observation."""
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def step(
        self, tool_calls: Union[List[EnvToolCall], EnvToolCall, Dict, List[Dict], str]
    ) -> InternalObservation:
        """Execute one step in the environment."""
        try:
            validated_call = self.validate_tool_calls(tool_calls)
        except ValueError as e:
            # Return current state with error
            priv, pub = self.engine.get_current_states_for_observation()
            return await self._to_observation(
                priv,
                pub,
                self.custom_step_observation_callable,
                extra_obs={"error": str(e)},
            )

        # Execute the tool
        result = await self._interact_tool(validated_call)

        if result.ok:
            priv = result.payload["private_state"]
            pub = result.payload["public_state"]
            return await self._to_observation(priv, pub, self.custom_step_observation_callable)
        else:
            # Tool failed - return error with current state
            priv, pub = self.engine.get_current_states_for_observation()
            return await self._to_observation(
                priv,
                pub,
                self.custom_step_observation_callable,
                extra_obs={"error": result.error},
            )

    async def checkpoint(self) -> InternalObservation:
        """Create a checkpoint observation."""
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    async def terminate(self) -> InternalObservation:
        """Terminate the environment."""
        priv, pub = self.engine.get_current_states_for_observation()

        # Mark as terminated
        pub.terminated = True
        priv.terminated = True

        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    def validate_tool_calls(
        self, tool_calls: Union[List[EnvToolCall], EnvToolCall, Dict, List[Dict], str]
    ) -> EnvToolCall:
        """Validate and normalize tool calls."""
        # Handle string input (simple action)
        if isinstance(tool_calls, str):
            return EnvToolCall(tool="interact", args={"action": tool_calls})

        # Handle dict input
        if isinstance(tool_calls, dict):
            # Check if it's already properly formatted
            if "tool" in tool_calls and "args" in tool_calls:
                # Handle tool name aliases
                tool_name = tool_calls["tool"]
                if tool_name == "nethack_interact":
                    tool_name = "interact"
                return EnvToolCall(tool=tool_name, args=tool_calls["args"])  # type: ignore[misc]
            elif "tool_name" in tool_calls and "args" in tool_calls:
                # Handle legacy format
                tool_name = tool_calls["tool_name"]
                if tool_name == "nethack_interact":
                    tool_name = "interact"
                return EnvToolCall(tool=tool_name, args=tool_calls["args"])
            # Check for action key
            elif "action" in tool_calls:
                return EnvToolCall(tool="interact", args={"action": tool_calls["action"]})
            # Check for tool_calls format
            elif "tool_calls" in tool_calls:
                tool_calls = tool_calls["tool_calls"]
                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                    return self.validate_tool_calls(tool_calls[0])
            # Try to extract action from various formats
            else:
                # Look for action in nested structures
                for key in ["args", "parameters", "input"]:
                    if key in tool_calls and isinstance(tool_calls[key], dict):
                        if "action" in tool_calls[key]:
                            return EnvToolCall(
                                tool="interact",
                                args={"action": tool_calls[key]["action"]},
                            )

        # Handle list input
        if isinstance(tool_calls, list):
            if len(tool_calls) == 0:
                raise ValueError("Empty tool calls list")
            # Take first tool call
            return self.validate_tool_calls(tool_calls[0])

        # Handle EnvToolCall object
        if isinstance(tool_calls, EnvToolCall):
            return tool_calls

        raise ValueError(
            f"Invalid tool call format. Expected action string, dict with 'action' key, "
            f"or EnvToolCall object. Got: {type(tool_calls)}"
        )

    async def _to_observation(
        self,
        private_state: NetHackPrivateState,
        public_state: NetHackPublicState,
        observation_callable: GetObservationCallable,
        extra_obs: Optional[Dict[str, Any]] = None,
    ) -> InternalObservation:
        """Convert states to observation using the callable."""
        obs = await observation_callable.get_observation(public_state, private_state)  # type: ignore[call-arg]

        if extra_obs:
            obs.update(extra_obs)

        return obs

    async def _serialize_engine(self) -> Any:
        """Serialize the engine state."""
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, task_instance: TaskInstance, snapshot: Any) -> NetHackEngine:
        """Deserialize the engine from a snapshot."""
        return await NetHackEngine._deserialize_engine(snapshot)

    def get_available_actions(self) -> List[str]:
        """Get list of all available actions."""
        return list(NETHACK_ACTIONS.keys()) + list(MENU_ACTIONS.keys())

    def get_action_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all actions."""
        return {**NETHACK_ACTIONS, **MENU_ACTIONS}
