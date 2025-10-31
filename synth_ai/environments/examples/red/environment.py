from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import base64
import time
from io import BytesIO

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
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.tracing_v3.abstractions import EnvironmentEvent, TimeRecord
from synth_ai.tracing_v3.session_tracer import SessionTracer
try:  # optional for image encoding
    import numpy as _np  # type: ignore
    from PIL import Image as _PILImage  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _np = None  # type: ignore
    _PILImage = None  # type: ignore

# Import logging configuration to suppress JAX debug messages
from .engine import (
    PokemonRedEngine,
    PokemonRedEngineSnapshot,
    PokemonRedPrivateState,
    PokemonRedPublicState,
)
from .taskset import INSTANCE as DEFAULT_TASK_INSTANCE
from .taskset import PokemonRedTaskInstance


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
            format_hp_status,
            format_position,
            get_badge_count,
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
        tracer: Optional[SessionTracer] = None,
    ):
        self.name = "PokemonRed"
        self.task_instance = task_instance or DEFAULT_TASK_INSTANCE
        self.custom_step_observation_callable = custom_step_obs or PokemonRedObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or PokemonRedObservationCallable()
        )
        self.engine = PokemonRedEngine(self.task_instance)
        self.tracer = tracer

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

        # Record EnvironmentEvent for tracing if tracer is available
        if self.tracer and hasattr(priv_state, 'reward_last_step'):
            # Get state information for the event
            prev_state = getattr(self.engine, '_previous_state', None)
            terminated = getattr(priv_state, 'terminated', False)
            truncated = getattr(priv_state, 'truncated', False)

            # Convert states to dict for serialization
            pub_state_dict = pub_state.__dict__ if hasattr(pub_state, '__dict__') else pub_state

            env_event = EnvironmentEvent(
                system_instance_id="pokemon_red_env",
                time_record=TimeRecord(event_time=time.time()),
                reward=float(priv_state.reward_last_step),
                terminated=terminated,
                truncated=truncated,
                system_state_before=prev_state if prev_state else None,
                system_state_after=pub_state_dict,
            )
            await self.tracer.record_event(env_event)

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
        
        # Include raw state fields for reward calculation
        if isinstance(observation, dict):
            observation["map_id"] = pub.world.map_id if pub.world else None
            observation["player_x"] = pub.world.player_x if pub.world else None
            observation["player_y"] = pub.world.player_y if pub.world else None
            observation["party_count"] = len(pub.party) if pub.party else 0
            observation["party_pokemon"] = [
                {
                    "species_id": p.species_id,
                    "level": p.level,
                    "hp_current": p.hp_current,
                    "hp_max": p.hp_max,
                    "hp_percentage": (p.hp_current / p.hp_max * 100) if p.hp_max > 0 else 0,
                }
                for p in (pub.party or [])
            ]
            observation["in_battle"] = pub.system.in_battle if pub.system else False
            observation["battle_outcome"] = pub.system.battle_outcome if pub.system else 0
            observation["text_box_active"] = pub.system.text_box_active if pub.system else False
            observation["enemy_hp_current"] = pub.system.enemy_hp_current if pub.system else 0
            observation["enemy_hp_max"] = pub.system.enemy_hp_max if pub.system else 0
            observation["enemy_hp_percentage"] = pub.system.enemy_hp_percentage if pub.system else 0.0
            observation["badges"] = pub.progress.badges if pub.progress else 0
        # Attach latest PNG frame for VLM agents if available
        try:
            emulator = getattr(self.engine, "emulator", None)
            screen = getattr(emulator, "screen", None)
            if screen is not None and _np is not None and _PILImage is not None:
                # Prefer documented ndarray property if present
                frame = getattr(screen, "ndarray", None)
                if frame is None and hasattr(screen, "image"):
                    frame = screen.image
                if isinstance(frame, _np.ndarray) and frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
                    array_uint8 = (
                        frame.astype("uint8") if frame.dtype != _np.uint8 else frame
                    )
                    # PyBoy gives RGBA; convert to RGB
                    if array_uint8.shape[-1] == 4:
                        array_uint8 = array_uint8[:, :, :3]
                    img = _PILImage.fromarray(array_uint8, mode="RGB")
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                    if isinstance(observation, dict):
                        observation["observation_image_base64"] = encoded
                        observation["observation_image_format"] = "png"
                        observation["observation_image_width"] = int(array_uint8.shape[1])
                        observation["observation_image_height"] = int(array_uint8.shape[0])
                        observation["observation_image_data_url"] = f"data:image/png;base64,{encoded}"
        except Exception:
            pass
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
