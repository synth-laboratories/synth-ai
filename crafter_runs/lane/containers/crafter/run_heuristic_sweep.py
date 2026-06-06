from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any


ACTION_NAMES = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "make_wood_axe",
    "make_stone_axe",
    "make_iron_axe",
    "make_wood_wall",
    "make_stone_wall",
    "make_iron_wall",
    "make_wood_door",
    "make_stone_door",
    "make_iron_door",
]
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_policy(policy_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy_cls = getattr(module, "Policy", None)
    if policy_cls is None:
        raise RuntimeError("policy file must define Policy")
    return policy_cls()


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _inventory(env: Any, info: dict[str, Any]) -> dict[str, int]:
    player = getattr(env, "_player", None)
    raw = info.get("inventory")
    if raw is None and player is not None:
        raw = getattr(player, "inventory", {})
    result: dict[str, int] = {}
    for key, value in _as_mapping(raw).items():
        if isinstance(value, (int, float, bool)):
            result[str(key)] = int(value)
    return result


def _achievements(env: Any, info: dict[str, Any]) -> dict[str, bool]:
    player = getattr(env, "_player", None)
    raw = info.get("achievements")
    if raw is None and player is not None:
        raw = getattr(player, "achievements", {})
    return {str(key): bool(value) for key, value in _as_mapping(raw).items()}


def _observation(env: Any, info: dict[str, Any], *, reward: float, done: bool, step: int) -> dict[str, Any]:
    inventory = _inventory(env, info)
    achievements = _achievements(env, info)
    player = getattr(env, "_player", None)
    health = getattr(player, "health", None) if player is not None else None
    if not isinstance(health, (int, float)):
        health = inventory.get("health", 0)
    return {
        "reward": float(reward),
        "inventory": inventory,
        "health": float(health or 0),
        "achievements_status": achievements,
        "achievements": [name for name, unlocked in achievements.items() if unlocked],
        "done": bool(done),
        "step_count": int(step),
    }


def _action_id(action: Any) -> int:
    if isinstance(action, str):
        return int(ACTION_TO_ID.get(action, 0))
    if isinstance(action, (int, float)):
        return max(0, int(action))
    return 0


def _run_episode(*, policy: Any, seed: int, max_steps: int) -> dict[str, Any]:
    import crafter

    env = crafter.Env(seed=seed)
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        _obs, info = reset_out
    else:
        _obs, info = reset_out, {}
    info = info if isinstance(info, dict) else {}
    total_reward = 0.0
    done = False
    step = 0
    last_observation = _observation(env, info, reward=0.0, done=False, step=0)
    try:
        for step in range(1, max_steps + 1):
            action = _action_id(policy.act(last_observation, info))
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                _obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            else:
                _obs, reward, done, info = step_out
            info = info if isinstance(info, dict) else {}
            total_reward += float(reward or 0.0)
            last_observation = _observation(env, info, reward=float(reward or 0.0), done=done, step=step)
            if done:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "seed": seed,
        "status": "completed",
        "reward": round(total_reward, 4),
        "steps": step,
        "done": done,
        "health": last_observation.get("health"),
        "achievements": last_observation.get("achievements", []),
    }


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--seeds", default="101,103,105,107")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    policy_path = Path(args.policy_path).expanduser().resolve()
    policy = _load_policy(policy_path)
    seeds = _parse_seeds(args.seeds)
    rows = [_run_episode(policy=policy, seed=seed, max_steps=max(1, args.max_steps)) for seed in seeds]
    rewards = [float(row["reward"]) for row in rows]
    achievement_frequency: dict[str, int] = {}
    for row in rows:
        for name in row.get("achievements") or []:
            achievement_frequency[str(name)] = achievement_frequency.get(str(name), 0) + 1
    health_values = [float(row.get("health") or 0.0) for row in rows]
    summary = {
        "schema_version": "hillclimbsymbolicbench.crafter_summary.v1",
        "env": "crafter",
        "seed_count": len(seeds),
        "episode_count": len(rows),
        "completed": len(rows),
        "failed": 0,
        "reward": {
            "mean": round(statistics.mean(rewards), 4) if rewards else 0.0,
            "median": round(statistics.median(rewards), 4) if rewards else 0.0,
        },
        "health_mean": round(statistics.mean(health_values), 4) if health_values else 0.0,
        "achievement_frequency": achievement_frequency,
        "achievement_count_mean": round(
            statistics.mean(len(row.get("achievements") or []) for row in rows),
            4,
        )
        if rows
        else 0.0,
        "failure_modes": {},
    }
    _write_json(output_dir / "results.json", {"results": rows})
    _write_json(output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
