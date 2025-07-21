from __future__ import annotations
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field

# Import logging configuration to suppress JAX debug messages

from .engine import (
    PokemonRedEngine,
    PokemonRedPrivateState,
    PokemonRedPublicState,
    PokemonRedEngineSnapshot,
)
from .taskset import PokemonRedTaskInstance, INSTANCE as DEFAULT_TASK_INSTANCE
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


# Tool input schemas
class PressButtonInput(BaseModel):
    button: str = Field(
        ..., description="Game Boy button: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT"
    )
    frames: int = Field(1, description="Number of frames to hold the button")


# Tool definitions
class PressButtonTool(AbstractTool):
    name = "press_button"
    description = "Press a Game Boy button for the specified number of frames"
    call_schema = PressButtonInput
    result_schema = ToolResult

    def __init__(self, engine: PokemonRedEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            action = {"button": validated_args.button, "frames": validated_args.frames}
            priv_state, pub_state = await self.engine._step_engine(action)
            return ToolResult(
                ok=True,
                payload={
                    "public": pub_state,
                    "private": priv_state,
                },
            )
        except Exception as e:
            # Get current state for error context
            priv_state, pub_state = self.engine._create_states(reward=0.0)
            return ToolResult(
                ok=False,
                error=str(e),
                payload={"public": pub_state},
            )


# Observation callable for Pokemon Red
class PokemonRedObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: PokemonRedPublicState, priv: PokemonRedPrivateState
    ) -> InternalObservation:
        """Convert Pokemon Red states to agent observation"""
        from .engine_helpers.state_extraction import (
            get_badge_count,
            format_position,
            format_hp_status,
        )

        badge_count = get_badge_count(pub.badges)
        position = format_position(pub.player_x, pub.player_y, pub.map_id)
        hp_status = format_hp_status(pub.party_hp_current, pub.party_hp_max)

        obs = {
            "position": position,
            "badges_earned": badge_count,
            "badges_bitfield": pub.badges,
            "hp_status": hp_status,
            "party_level": pub.party_level,
            "party_xp": pub.party_xp,
            "in_battle": pub.in_battle,
            "step_count": pub.step_count,
            "reward_last_step": priv.reward_last_step,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
        }

        if pub.error_info:
            obs["error"] = pub.error_info

        return obs


class PokemonRedEnvironment(StatefulEnvironment, ReproducibleEnvironment[PokemonRedEngine]):
    """Pokemon Red stateful game environment for AI agents"""

    def __init__(
        self,
        task_instance: Optional[PokemonRedTaskInstance] = None,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "PokemonRed"
        self.task_instance = task_instance or DEFAULT_TASK_INSTANCE
        self.custom_step_observation_callable = custom_step_obs or PokemonRedObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or PokemonRedObservationCallable()
        )
        self.engine = PokemonRedEngine(self.task_instance)

        # Register tools
        self._press_button_tool = PressButtonTool(self.engine)
        if self._press_button_tool.name not in TOOL_REGISTRY:
            register_tool(self._press_button_tool)

    async def initialize(self) -> InternalObservation:
        """Initialize the Pokemon Red environment"""
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:
        """Terminate the environment"""
        priv, pub = self.engine._create_states(reward=0.0, terminated=True)
        obs_dict = {
            "terminated": True,
            "message": "Pokemon Red environment terminated.",
        }
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable, extra_obs=obs_dict
        )

    def validate_tool_calls(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> EnvToolCall:
        """Validate and normalize tool calls to single EnvToolCall"""
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
        if agent_call.tool != "press_button":
            raise ValueError(f"Unknown tool: {agent_call.tool}. Expected 'press_button'.")

        return agent_call

    async def step(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> InternalObservation:
        """Execute one step in the Pokemon Red environment"""
        agent_call = self.validate_tool_calls(tool_calls)
        tool_result: ToolResult = await self._press_button_tool(agent_call)

        payload_dict = tool_result.payload
        if not tool_result.ok or not isinstance(payload_dict, dict):
            # Fallback if tool execution failed
            priv_state, pub_state = self.engine._create_states(reward=0.0)
            if tool_result.error and hasattr(pub_state, "error_info"):
                pub_state.error_info = tool_result.error
        else:
            # Extract states from successful tool execution - now they're dataclass objects
            priv_state = payload_dict.get("private")
            pub_state = payload_dict.get("public")

            if priv_state is None or pub_state is None:
                priv_state, pub_state = self.engine._create_states(reward=0.0)
                if tool_result.error and hasattr(pub_state, "error_info"):
                    pub_state.error_info = tool_result.error
            else:
                # States are already dataclass objects, no need to reconstruct
                if tool_result.error and hasattr(pub_state, "error_info"):
                    pub_state.error_info = tool_result.error

        return await self._to_observation(
            priv_state, pub_state, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:
        """Create a checkpoint of the current environment state"""
        engine_snapshot: PokemonRedEngineSnapshot = await self.engine._serialize_engine()
        priv, pub = self.engine._create_states(reward=0.0)
        obs_data = await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )
        if isinstance(obs_data, dict):
            obs_data["engine_snapshot_data"] = engine_snapshot.model_dump()
        return obs_data

    async def _to_observation(
        self,
        priv: PokemonRedPrivateState,
        pub: PokemonRedPublicState,
        obs_cb: Optional[GetObservationCallable],
        extra_obs: Optional[Dict[str, Any]] = None,
    ) -> InternalObservation:
        """Convert states to observation using the specified callback"""
        active_obs_cb = obs_cb or PokemonRedObservationCallable()
        observation = await active_obs_cb.get_observation(pub, priv)
        if extra_obs and isinstance(observation, dict):
            observation.update(extra_obs)
        return observation

    # ReproducibleEnvironment methods
    async def _serialize_engine(self) -> PokemonRedEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: PokemonRedEngineSnapshot, task_instance: PokemonRedTaskInstance
    ) -> "PokemonRedEnvironment":
        eng = await PokemonRedEngine._deserialize_engine(snapshot, task_instance)
        env = cls(task_instance)
        env.engine = eng
        return env
