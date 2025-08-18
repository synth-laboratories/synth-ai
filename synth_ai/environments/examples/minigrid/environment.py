"""MiniGrid Environment implementation.

This module provides a high-level interface for MiniGrid environments
with tool-based interaction and flexible observation generation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.environment.tools import AbstractTool, EnvToolCall, ToolResult
from synth_ai.environments.examples.minigrid.engine import (
    MiniGridCheckpointObservationCallable,
    MiniGridEngine,
    MiniGridObservationCallable,
    MiniGridPrivateState,
    MiniGridPublicState,
)
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.tasks.core import TaskInstance


class MiniGridActionInput(BaseModel):
    """Input model for MiniGrid actions."""

    action: str = Field(
        ...,
        description="The action to take. Must be one of: 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'",
    )


class MiniGridInteractTool(AbstractTool):
    """Tool for interacting with the MiniGrid environment."""

    name = "minigrid_act"
    description = "Perform an action in the MiniGrid environment"
    call_schema = MiniGridActionInput
    result_schema = ToolResult

    def __init__(self, engine: MiniGridEngine):
        """Initialize the tool with a MiniGrid engine."""
        self.engine = engine
        self.action_map = {
            "left": 0,  # Action 0 is counter-clockwise (left)
            "right": 1,  # Action 1 is clockwise (right)
            "forward": 2,
            "pickup": 3,
            "drop": 4,
            "toggle": 5,
            "done": 6,
        }

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        """Execute the action and return the result."""
        try:
            action_name = call.args.get("action", "").lower()

            if action_name not in self.action_map:
                return ToolResult(
                    ok=False,
                    error=f"Invalid action '{action_name}'. Valid actions are: {', '.join(self.action_map.keys())}",
                    payload={},
                )

            action = self.action_map[action_name]

            # Execute the action
            private_state, public_state = await self.engine._step_engine(action)

            # Build response
            response_parts = [f"Action '{action_name}' executed."]

            if private_state.reward_last != 0:
                response_parts.append(f"Reward: {private_state.reward_last:.2f}")

            if private_state.terminated:
                response_parts.append("Episode terminated!")
                if private_state.info.get("success", False):
                    response_parts.append("Mission completed successfully!")
            elif private_state.truncated:
                response_parts.append("Episode truncated (max steps reached).")

            return ToolResult(
                ok=True,
                payload={
                    "message": " ".join(response_parts),
                    "public_state": public_state,
                    "private_state": private_state,
                },
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e), payload={})


class MiniGridEnvironment(StatefulEnvironment, ReproducibleEnvironment[MiniGridEngine]):
    """High-level MiniGrid environment with tool-based interaction."""

    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        """Initialize the MiniGrid environment.

        Args:
            task_instance: Task instance containing configuration
            custom_step_obs: Custom observation generator for steps
            custom_ckpt_obs: Custom observation generator for checkpoints
        """
        self.name = "MiniGridEnvironment"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or MiniGridObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or MiniGridCheckpointObservationCallable()
        )

        # Create engine
        self.engine = MiniGridEngine(task_instance)

        # Initialize tool
        self._interact_tool = MiniGridInteractTool(self.engine)

    async def initialize(self) -> InternalObservation:
        """Initialize the environment and return initial observation."""
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def step(
        self,
        tool_calls: Union[List[Dict[str, Any]], List[EnvToolCall], Dict[str, Any], EnvToolCall],
    ) -> InternalObservation:
        """Process a tool call and return observation."""
        validated_call = self.validate_tool_calls(tool_calls)
        result = await self._interact_tool(validated_call)

        if result.ok:
            priv = result.payload["private_state"]
            pub = result.payload["public_state"]
            return await self._to_observation(priv, pub, self.custom_step_observation_callable)
        else:
            priv, pub = self.engine.get_current_states_for_observation()
            return await self._to_observation(
                priv,
                pub,
                self.custom_step_observation_callable,
                extra_obs={"error": result.error},
            )

    async def terminate(self) -> InternalObservation:
        """Terminate the environment and return final observation."""
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    async def checkpoint(self) -> InternalObservation:
        """Create a checkpoint of the current state."""
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    def validate_tool_calls(
        self,
        tool_calls: Union[List[Dict[str, Any]], List[EnvToolCall], Dict[str, Any], EnvToolCall],
    ) -> EnvToolCall:
        """Validate and normalize tool calls."""
        # If already an EnvToolCall, validate and return
        if isinstance(tool_calls, EnvToolCall):
            if tool_calls.tool != "minigrid_act":
                raise ValueError(f"Unknown tool: {tool_calls.tool}. Expected 'minigrid_act'.")
            return tool_calls

        # Handle different input formats
        if isinstance(tool_calls, dict):
            # Single tool call
            tool_call = tool_calls
        elif isinstance(tool_calls, list) and len(tool_calls) > 0:
            # List of tool calls - take the first one
            first_item = tool_calls[0]
            if isinstance(first_item, list) and len(first_item) > 0:
                # Nested list
                tool_call = first_item[0]
            elif isinstance(first_item, EnvToolCall):
                # Handle case where service sends list of EnvToolCall objects
                if first_item.tool != "minigrid_act":
                    raise ValueError(f"Unknown tool: {first_item.tool}. Expected 'minigrid_act'.")
                return first_item
            else:
                tool_call = first_item
        else:
            raise ValueError("Invalid tool_calls format")

        # At this point tool_call should be a dict
        if isinstance(tool_call, EnvToolCall):
            # Handle case where we somehow still have an EnvToolCall
            if tool_call.tool != "minigrid_act":
                raise ValueError(f"Unknown tool: {tool_call.tool}. Expected 'minigrid_act'.")
            return tool_call

        # Extract tool name and args - fail fast
        if "tool" in tool_call:
            tool_name = tool_call["tool"]
        elif "name" in tool_call:
            tool_name = tool_call["name"]
        else:
            raise ValueError("Tool call missing 'tool' or 'name' field")

        # Handle different argument formats - fail fast
        if "args" in tool_call:
            args = tool_call["args"]
        elif "parameters" in tool_call:
            args = tool_call["parameters"]
        elif "input" in tool_call:
            if isinstance(tool_call["input"], str):
                args = json.loads(tool_call["input"])
            else:
                args = tool_call["input"]
        else:
            raise ValueError("Tool call missing 'args', 'parameters', or 'input' field")

        if tool_name != "minigrid_act":
            raise ValueError(f"Unknown tool: {tool_name}. Expected 'minigrid_act'.")

        # Create EnvToolCall
        return EnvToolCall(
            tool=tool_name,
            args=args,
        )

    async def _to_observation(
        self,
        priv: MiniGridPrivateState,
        pub: MiniGridPublicState,
        observation_callable: GetObservationCallable,
        extra_obs: Optional[Dict[str, Any]] = None,
    ) -> InternalObservation:
        """Convert states to observation using callable."""
        obs = await observation_callable.get_observation(pub, priv)

        # Attach full state objects for downstream analysis (fail fast)
        obs["public"] = pub
        obs["private"] = priv

        if extra_obs:
            obs.update(extra_obs)

        return obs

    async def _serialize_engine(self) -> Dict[str, Any]:
        """Serialize the engine state."""
        snapshot = await self.engine._serialize_engine()
        return {
            "task_instance_dict": snapshot.task_instance_dict,
            "engine_snapshot": snapshot.engine_snapshot,
        }

    @classmethod
    async def _deserialize_engine(cls, data: Dict[str, Any]) -> MiniGridEngine:
        """Deserialize the engine state."""
        from synth_ai.environments.examples.minigrid.engine import MiniGridEngineSnapshot

        snapshot = MiniGridEngineSnapshot(
            task_instance_dict=data["task_instance_dict"],
            engine_snapshot=data["engine_snapshot"],
        )

        return await MiniGridEngine._deserialize_engine(snapshot)
