from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

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
    BanditEngine,
    BanditEngineSnapshot,
    SynthBanditCheckpointObservationCallable,
    SynthBanditObservationCallable,
)


class BanditActionInput(BaseModel):
    arm: int = Field(..., ge=0, description="Index of the arm to pull (0-based)")


class BanditInteractTool(AbstractTool):
    name = "pull_arm"
    description = "Pull a specific bandit arm to receive a stochastic reward."
    call_schema = BanditActionInput
    result_schema = ToolResult

    def __init__(self, engine: BanditEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            parsed = self.call_schema(**call.args)
        except ValidationError as exc:
            return ToolResult(ok=False, error=str(exc))

        try:
            private_state, public_state = await self.engine._step_engine(parsed.arm)
        except Exception as exc:  # noqa: BLE001 - propagate as tool error
            priv, pub = self.engine.get_current_states_for_observation()
            return ToolResult(
                ok=False,
                error=str(exc),
                payload={"private_state": priv, "public_state": pub},
            )

        return ToolResult(
            ok=True,
            payload={"private_state": private_state, "public_state": public_state},
        )


class BanditEnvironment(StatefulEnvironment, ReproducibleEnvironment[BanditEngine]):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: GetObservationCallable | None = None,
        custom_ckpt_obs: GetObservationCallable | None = None,
    ) -> None:
        self.name = "Bandit"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or SynthBanditObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthBanditCheckpointObservationCallable()
        )
        self.engine = BanditEngine(task_instance)
        self._interact_tool = BanditInteractTool(self.engine)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def step(self, tool_calls: list[EnvToolCall] | EnvToolCall | Any) -> InternalObservation:
        validated_call = self.validate_tool_calls(tool_calls)
        result = await self._interact_tool(validated_call)

        if result.ok and result.payload:
            priv = result.payload["private_state"]
            pub = result.payload["public_state"]
            return await self._to_observation(priv, pub, self.custom_step_observation_callable)

        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(
            priv,
            pub,
            self.custom_step_observation_callable,
            extra_obs={"error": result.error} if result.error else None,
        )

    async def checkpoint(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    async def terminate(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        self.engine.terminated = True
        pub.terminated = True
        priv.terminated = True
        if pub.status != "completed":
            pub.status = "terminated"
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    def validate_tool_calls(self, tool_calls: Any) -> EnvToolCall:
        if isinstance(tool_calls, EnvToolCall):
            candidate = tool_calls
        elif isinstance(tool_calls, dict):
            if "tool" in tool_calls:
                candidate = EnvToolCall(tool=tool_calls["tool"], args=tool_calls.get("args", {}))
            elif "name" in tool_calls:
                candidate = EnvToolCall(
                    tool=tool_calls["name"], args=tool_calls.get("parameters", {})
                )
            elif "function" in tool_calls:
                func_spec = tool_calls["function"]
                candidate = EnvToolCall(
                    tool=func_spec.get("name", "pull_arm"),
                    args=func_spec.get("arguments", {}),
                )
            else:
                candidate = EnvToolCall(tool="pull_arm", args=tool_calls)
        elif isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Empty tool calls list")
            return self.validate_tool_calls(tool_calls[0])
        else:
            if isinstance(tool_calls, int):
                return EnvToolCall(tool="pull_arm", args={"arm": tool_calls})
            try:
                return EnvToolCall(tool="pull_arm", args={"arm": int(tool_calls)})
            except (TypeError, ValueError) as exc:
                raise TypeError("Unsupported tool_calls format for bandit environment") from exc

        tool_name = candidate.tool
        if tool_name not in {"pull_arm", "interact"}:
            raise ValueError(f"Unknown tool: {tool_name}")

        args = dict(candidate.args or {})
        if "arm" not in args:
            for alias in ("action", "index", "arm_index", "armId", "choice"):
                if alias in args:
                    args["arm"] = args[alias]
                    break
        if "arm" not in args:
            raise ValueError("Missing required 'arm' argument")

        try:
            arm_idx = int(args["arm"])
        except (TypeError, ValueError) as exc:
            raise ValueError("'arm' argument must be an integer") from exc

        if arm_idx < 0:
            raise ValueError("'arm' argument must be non-negative")

        return EnvToolCall(tool="pull_arm", args={"arm": arm_idx})

    async def _to_observation(
        self,
        priv,
        pub,
        obs_cb: GetObservationCallable | None,
        extra_obs: dict[str, Any] | None = None,
    ) -> InternalObservation:
        observation: InternalObservation
        if obs_cb:
            observation = await obs_cb.get_observation(pub, priv)
        else:
            observation = {}
        if extra_obs and isinstance(observation, dict):
            observation.update(extra_obs)
        return observation

    async def _serialize_engine(self) -> BanditEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls,
        snapshot: BanditEngineSnapshot,
        task_instance: TaskInstance,
    ) -> BanditEnvironment:
        env = cls(task_instance)
        env.engine = await BanditEngine._deserialize_engine(snapshot)
        env._interact_tool = BanditInteractTool(env.engine)
        return env
