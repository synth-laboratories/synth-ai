from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
)
from synth_ai.environments.reproducibility.core import ReproducibleEnvironment
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.tasks.core import TaskInstance

from .engine import (
    SynthWordleCheckpointObservationCallable,
    SynthWordleObservationCallable,
    WordleEngine,
    WordleEngineSnapshot,
    WordlePrivateState,
    WordlePublicState,
)


class WordleActionInput(BaseModel):
    guess: str = Field(..., description="Your word guess (letters only)")


class WordleInteractTool(AbstractTool):
    name = "interact"
    description = "Submit a word guess to the Wordle environment."
    call_schema = WordleActionInput
    result_schema = ToolResult

    def __init__(self, engine: WordleEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated = self.call_schema(**call.args)
            priv, pub = await self.engine._step_engine(validated.guess)
            return ToolResult(ok=True, payload={"public_state": pub, "private_state": priv})
        except Exception as e:
            # Return current state with error message
            priv, pub = self.engine.get_current_states_for_observation()
            return ToolResult(
                ok=False, error=str(e), payload={"public_state": pub, "private_state": priv}
            )


class WordleEnvironment(StatefulEnvironment, ReproducibleEnvironment[WordleEngine]):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: GetObservationCallable | None = None,
        custom_ckpt_obs: GetObservationCallable | None = None,
    ) -> None:
        self.name = "Wordle"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or SynthWordleObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthWordleCheckpointObservationCallable()
        )
        self.engine = WordleEngine(task_instance)
        self._interact_tool = WordleInteractTool(self.engine)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def step(self, tool_calls) -> InternalObservation:
        validated_call = self.validate_tool_calls(tool_calls)
        result = await self._interact_tool(validated_call)
        if result.ok:
            priv = result.payload["private_state"]
            pub = result.payload["public_state"]
            return await self._to_observation(priv, pub, self.custom_step_observation_callable)
        else:
            priv, pub = self.engine.get_current_states_for_observation()
            return await self._to_observation(
                priv, pub, self.custom_step_observation_callable, extra_obs={"error": result.error}
            )

    async def checkpoint(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    async def terminate(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        pub.terminated = True
        priv.terminated = True
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    def validate_tool_calls(self, tool_calls) -> EnvToolCall:
        # Accept EnvToolCall, dict-like, or list formats similar to other envs
        if isinstance(tool_calls, EnvToolCall):
            validated = tool_calls
        elif isinstance(tool_calls, dict):
            if "tool" in tool_calls:
                validated = EnvToolCall(tool=tool_calls["tool"], args=tool_calls.get("args", {}))
            elif "name" in tool_calls:
                validated = EnvToolCall(
                    tool=tool_calls["name"], args=tool_calls.get("parameters", {})
                )
            elif "function" in tool_calls:
                validated = EnvToolCall(
                    tool=tool_calls["function"]["name"],
                    args=tool_calls["function"].get("arguments", {}),
                )
            else:
                # Treat remaining keys as args; default tool name
                validated = EnvToolCall(tool="interact", args=tool_calls)
        elif isinstance(tool_calls, list):
            if len(tool_calls) == 0:
                raise ValueError("Empty tool calls list")
            validated = self.validate_tool_calls(tool_calls[0])
        else:
            # Assume it's a raw guess string
            validated = EnvToolCall(tool="interact", args={"guess": str(tool_calls)})

        if validated.tool != "interact":
            raise ValueError(f"Unknown tool: {validated.tool}")
        # Normalize: allow 'action' key synonymous with 'guess'
        args = validated.args
        if "action" in args and "guess" not in args:
            args = {"guess": args["action"]}
        return EnvToolCall(tool="interact", args=args)

    async def _to_observation(
        self,
        priv: WordlePrivateState,
        pub: WordlePublicState,
        obs_cb: GetObservationCallable | None,
        extra_obs: dict[str, Any] | None = None,
    ) -> InternalObservation:
        if obs_cb:
            obs = await obs_cb.get_observation(pub, priv)
        else:
            obs: InternalObservation = {}
        if extra_obs and isinstance(obs, dict):
            obs.update(extra_obs)
        return obs

    async def _serialize_engine(self) -> WordleEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: WordleEngineSnapshot, task_instance: TaskInstance
    ) -> WordleEnvironment:
        env = cls(task_instance)
        env.engine = await WordleEngine._deserialize_engine(snapshot)
        env._interact_tool = WordleInteractTool(env.engine)
        return env
