from __future__ import annotations

from typing import Any, Dict, Tuple
import uuid

from synth_ai.environments.examples.crafter_classic.environment import (
    CrafterClassicEnvironment,
)
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.environments.environment.tools import EnvToolCall


class EnvRegistry:
    def __init__(self) -> None:
        self._envs: Dict[str, CrafterClassicEnvironment] = {}

    async def initialize(self, config: Dict[str, Any] | None) -> Tuple[str, Dict[str, Any]]:
        cfg = dict(config or {})
        seed = int(cfg.get("seed", 0)) if cfg.get("seed") is not None else 0

        metadata = CrafterTaskInstanceMetadata(
            difficulty="normal",
            seed=seed,
            num_trees_radius=0,
            num_cows_radius=0,
            num_hostiles_radius=0,
        )
        task = CrafterTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(instructions="Local prompt inspection"),
            intent=Intent(rubric={"goal": "none"}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        env = CrafterClassicEnvironment(task)
        env_id = f"env_{uuid.uuid4().hex[:8]}"
        self._envs[env_id] = env
        obs = await env.initialize(seed=seed)
        return env_id, obs

    async def step(self, env_id: str, action_name: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        env = self._envs[env_id]
        action_map = {
            "noop": 0,
            "move_left": 1,
            "move_right": 2,
            "move_up": 3,
            "move_down": 4,
            "do": 5,
            "sleep": 6,
        }
        a = int(action_map.get(action_name, 0))
        call = EnvToolCall(tool="interact", args={"action": a})
        obs = await env.step([call])
        reward = float(obs.get("reward_last_step", 0.0)) if isinstance(obs, dict) else 0.0
        done = bool(obs.get("terminated", False)) if isinstance(obs, dict) else False
        return obs, reward, done, {}

    async def terminate(self, env_id: str) -> Dict[str, Any]:
        env = self._envs.pop(env_id, None)
        if env is not None:
            try:
                await env.terminate()
            except Exception:
                pass
        return {"ok": True}


