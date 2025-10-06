from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import pytest


# Ensure examples/rl is importable so we can import crafter_task_app_helpers
_EXAMPLES_RL_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLES_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_RL_DIR))

from crafter_task_app_helpers.env import EnvRegistry  # type: ignore  # noqa: E402


def _get_position(obs: dict[str, Any]) -> Optional[Tuple[int, int]]:
    pos = obs.get("player_position") if isinstance(obs, dict) else None
    if pos is None:
        return None
    try:
        return int(pos[0]), int(pos[1])
    except Exception:
        return None


def _get_steps(obs: dict[str, Any]) -> int:
    try:
        return int(obs.get("num_steps_taken", 0)) if isinstance(obs, dict) else 0
    except Exception:
        return 0


@pytest.mark.asyncio
async def test_sequential_instances_are_fresh_and_isolated() -> None:
    reg = EnvRegistry()

    seed = 123
    env1_id, obs1_initial = await reg.initialize({"seed": seed})
    assert isinstance(env1_id, str) and env1_id, "env1_id must be a non-empty string"

    pos1_initial = _get_position(obs1_initial)
    steps1_initial = _get_steps(obs1_initial)
    assert steps1_initial == 0, f"expected fresh env to start at 0 steps, got {steps1_initial}"

    # Drive env1 forward a few steps
    last_obs_env1 = obs1_initial
    for _ in range(5):
        last_obs_env1, reward, done, _info = await reg.step(env1_id, "move_up")
        assert isinstance(reward, float), "reward must be float"
        assert isinstance(done, bool), "done must be bool"

    assert _get_steps(last_obs_env1) == 5, "env1 should have advanced exactly 5 steps"

    # Terminate env1 and start a new rollout with the same seed
    term1 = await reg.terminate(env1_id)
    assert term1.get("ok") is True, "terminate(env1) did not report ok=True"

    env2_id, obs2_initial = await reg.initialize({"seed": seed})
    assert isinstance(env2_id, str) and env2_id, "env2_id must be a non-empty string"
    assert env2_id != env1_id, "new rollout must allocate a distinct env_id"

    # New env must start fresh
    assert _get_steps(obs2_initial) == 0, "new rollout should start at 0 steps"

    # With identical seed, initial positions should match (reproducibility)
    pos2_initial = _get_position(obs2_initial)
    assert (
        pos1_initial == pos2_initial
    ), f"initial positions with same seed must match: {pos1_initial} vs {pos2_initial}"

    # Stepping a terminated env must fail loudly (no ghost instance)
    with pytest.raises(KeyError):
        await reg.step(env1_id, "move_left")


@pytest.mark.asyncio
async def test_concurrent_envs_do_not_cross_talk() -> None:
    reg = EnvRegistry()

    env_a, obs_a0 = await reg.initialize({"seed": 1})
    env_b, obs_b0 = await reg.initialize({"seed": 2})

    assert env_a != env_b, "concurrent rollouts must use different env_ids"
    assert _get_steps(obs_a0) == 0 and _get_steps(obs_b0) == 0, "both envs should start at 0 steps"

    # Interleave steps and assert isolation at each point
    obs_a, r_a, d_a, _ = await reg.step(env_a, "move_up")
    assert _get_steps(obs_a) == 1, "env_a should have advanced to 1 step"
    # env_b must remain unchanged until we step it
    assert _get_steps(obs_b0) == 0, "env_b should not advance when stepping env_a"

    obs_b, r_b, d_b, _ = await reg.step(env_b, "move_down")
    assert _get_steps(obs_b) == 1, "env_b should have advanced to 1 step"

    # More interleaving
    obs_a, r_a, d_a, _ = await reg.step(env_a, "move_up")
    obs_b, r_b, d_b, _ = await reg.step(env_b, "move_down")
    obs_a, r_a, d_a, _ = await reg.step(env_a, "move_up")
    obs_b, r_b, d_b, _ = await reg.step(env_b, "move_down")

    assert _get_steps(obs_a) == 3, f"env_a steps mismatch: {_get_steps(obs_a)}"
    assert _get_steps(obs_b) == 3, f"env_b steps mismatch: {_get_steps(obs_b)}"

    pos_a = _get_position(obs_a)
    pos_b = _get_position(obs_b)
    assert pos_a != pos_b, f"positions should diverge across isolated envs: {pos_a} vs {pos_b}"

    # Cleanup
    term_a = await reg.terminate(env_a)
    term_b = await reg.terminate(env_b)
    assert term_a.get("ok") and term_b.get("ok"), "terminate did not report ok for both envs"


