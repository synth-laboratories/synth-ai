from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel
import dataclasses

from synth_ai.environments.examples.sokoban.engine import (
    SokobanEngine,
    SynthSokobanObservationCallable,
    SokobanPrivateState,
    SokobanPublicState,
    SynthSokobanCheckpointObservationCallable,
    SokobanEngineSnapshot,
)
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.tasks.core import TaskInstance
from synth_ai.environments.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
    TOOL_REGISTRY,
    register_tool,
)


# --- Tool Definition ---
class SokobanActionInput(BaseModel):
    action: int


class SokobanInteractTool(AbstractTool):
    name = "interact"
    description = "Performs an action (e.g., move) in the Sokoban environment."
    call_schema = SokobanActionInput
    result_schema = ToolResult

    def __init__(self, engine: SokobanEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            priv_state, pub_state = await self.engine._step_engine(validated_args.action)
            return ToolResult(
                ok=True,
                payload={
                    "public": pub_state.to_dict(),
                    "private": priv_state.to_dict(),
                },
            )
        except Exception as e:
            # Add current public state to payload for context in case of error
            _, pub_state_on_error = self.engine.get_current_states_for_observation()
            return ToolResult(
                ok=False,
                error=str(e),
                payload={"public": pub_state_on_error.to_dict()},
            )


class SokobanEnvironment(StatefulEnvironment, ReproducibleEnvironment[SokobanEngine]):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "Sokoban"
        self.task_instance = task_instance
        # Default to SynthSokobanObservationCallable if none provided
        self.custom_step_observation_callable = custom_step_obs or SynthSokobanObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthSokobanCheckpointObservationCallable()
        )
        self.engine: SokobanEngine = SokobanEngine(task_instance)

        self._interact_tool = SokobanInteractTool(self.engine)
        if self._interact_tool.name not in TOOL_REGISTRY:
            register_tool(self._interact_tool)
        # elif getattr(TOOL_REGISTRY[self._interact_tool.name], 'engine', None) is not self.engine:
        # register_tool(self._interact_tool) # More robust check if tool has engine attr

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        priv.terminated = True  # Mark as terminated
        obs_dict = {"terminated": True, "message": "Environment terminated."}
        # Use _to_observation to format, including final state
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable, extra_obs=obs_dict
        )

    def validate_tool_calls(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> EnvToolCall:
        # Normalize and validate to a single EnvToolCall
        raw_call_data: Dict[str, Any]
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Received empty list of tool calls.")
            first_item = tool_calls[0]
            if isinstance(first_item, list):
                if not first_item:
                    raise ValueError("Received empty inner list of tool calls.")
                raw_call_data = first_item[0]
            elif isinstance(first_item, dict):
                raw_call_data = first_item
            elif isinstance(first_item, EnvToolCall):  # Already an EnvToolCall instance
                agent_call = first_item  # Assuming direct single call if already instance
                if agent_call.tool != "interact":
                    raise ValueError(f"Unknown tool: {agent_call.tool}. Expected 'interact'.")
                return agent_call
            else:
                raise TypeError(f"Unexpected type in tool_calls list: {type(first_item)}")
        elif isinstance(tool_calls, dict):  # Single call passed as dict
            raw_call_data = tool_calls
        elif isinstance(tool_calls, EnvToolCall):  # Single call already an instance
            if tool_calls.tool != "interact":
                raise ValueError(f"Unknown tool: {tool_calls.tool}. Expected 'interact'.")
            return tool_calls
        else:
            raise TypeError(f"Unexpected type for tool_calls: {type(tool_calls)}")

        if not isinstance(raw_call_data, dict):
            raise TypeError(f"Processed call data is not a dict: {type(raw_call_data)}")

        # Convert dict to EnvToolCall instance
        tool_name = raw_call_data.get("tool")
        tool_args = raw_call_data.get("args", {})
        if tool_name != "interact":
            raise ValueError(f"Unknown tool: {tool_name}. Expected 'interact'.")

        agent_call = EnvToolCall(tool=tool_name, args=tool_args)
        return agent_call

    async def step(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> InternalObservation:
        agent_call = self.validate_tool_calls(tool_calls)
        tool_result: ToolResult = await self._interact_tool(agent_call)

        payload_dict = tool_result.payload
        if not tool_result.ok or not isinstance(payload_dict, dict):  # Check tool_result.ok
            # Fallback if payload isn't as expected or tool reported an error
            priv_state, pub_state = self.engine.get_current_states_for_observation()
            if tool_result.error and hasattr(pub_state, "error_info"):
                pub_state.error_info = tool_result.error
        else:
            # This block assumes tool_result.ok is True and payload is a dict
            priv_dict = payload_dict.get("private")
            pub_dict = payload_dict.get("public")

            if priv_dict is None or pub_dict is None:
                # This case should ideally not happen if tool_result.ok is True
                # and the tool is well-behaved, but as a safeguard:
                priv_state, pub_state = self.engine.get_current_states_for_observation()
                if tool_result.error and hasattr(
                    pub_state, "error_info"
                ):  # Apply error even in this sub-optimal case
                    pub_state.error_info = tool_result.error
            else:
                priv_state = SokobanPrivateState(**priv_dict)
                pub_state = SokobanPublicState(**pub_dict)
                if tool_result.error and hasattr(pub_state, "error_info"):
                    pub_state.error_info = tool_result.error

        return await self._to_observation(
            priv_state, pub_state, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:
        engine_snapshot: SokobanEngineSnapshot = await self.engine._serialize_engine()
        # For checkpoint, we might want to convey the snapshot data differently.
        # The existing _to_observation expects live priv/pub states.
        # For now, using current live states for observation, plus snapshot.
        priv, pub = self.engine.get_current_states_for_observation()
        obs_data = await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )
        if isinstance(obs_data, dict):
            obs_data["engine_snapshot_data"] = (
                engine_snapshot.model_dump()
            )  # Add snapshot if obs is dict
        return obs_data

    async def _to_observation(
        self,
        priv: SokobanPrivateState,
        pub: SokobanPublicState,
        obs_cb: Optional[GetObservationCallable],
        extra_obs: Optional[Dict[str, Any]] = None,  # For adding things like termination messages
    ) -> InternalObservation:
        # Ensure obs_cb is not None; use a default if necessary (though __init__ sets one)
        active_obs_cb = obs_cb or SynthSokobanObservationCallable()
        observation = await active_obs_cb.get_observation(pub, priv)
        if extra_obs and isinstance(observation, dict):
            observation.update(extra_obs)
        return observation

    async def _serialize_engine(self) -> SokobanEngineSnapshot:  # Changed type hint
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: SokobanEngineSnapshot, task_instance: TaskInstance
    ) -> "SokobanEnvironment":  # Changed type hint
        eng = await SokobanEngine._deserialize_engine(snapshot, task_instance)
        env = cls(task_instance)  # Uses task_instance from deserialized engine
        env.engine = eng
        return env
