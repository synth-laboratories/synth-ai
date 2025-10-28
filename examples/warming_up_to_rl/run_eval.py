#!/usr/bin/env python3
"""
Baseline evaluation script (public-friendly skeleton)
- Targets a task app (Crafter-like) via initialize/step/terminate
- Uses a TaskAppClient interface (to be implemented in synth-ai SDK)
- Keeps structure aligned with research/testing/crafter eval harness
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib

import httpx


class TaskAppClient:
    """Minimal async client for the task app initialize/step/terminate routes.

    This is a public-friendly shim for examples, pending SDK surface consolidation.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TaskAppClient:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=600.0, follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Fallback for direct use without context manager
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=headers, timeout=600.0, follow_redirects=True
            )
        return self._client

    async def initialize(self, env_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """POST /env/{env_name}/initialize (compat route supported in task app)."""
        payload: dict[str, Any] = {
            "seed": config.get("seed"),
        }
        # Allow both world_config and config inputs; env routes will normalize difficulty
        if "world_config" in config:
            payload["world_config"] = config["world_config"]
        if "config" in config:
            payload["config"] = config["config"]
        resp = await self.client.post(f"/env/{env_name}/initialize", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def step(
        self, env_name: str, env_id: str, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """POST /env/{env_name}/step with wrapped tool_calls in action."""
        payload = {"env_id": env_id, "action": {"tool_calls": tool_calls}}
        resp = await self.client.post(f"/env/{env_name}/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def terminate(self, env_name: str, env_id: str) -> dict[str, Any]:
        resp = await self.client.post(f"/env/{env_name}/terminate", json={"env_id": env_id})
        resp.raise_for_status()
        return resp.json()

    async def get_info(self) -> dict[str, Any]:
        resp = await self.client.get("/info")
        resp.raise_for_status()
        return resp.json()

    async def proxy_groq_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/proxy/groq/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def vllm_chat(self, vllm_base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=vllm_base_url.rstrip("/"), timeout=60.0) as c:
            resp = await c.post("/v1/chat/completions", json=payload)
            # Do not raise for status to surface body in errors
            try:
                data = resp.json()
            except Exception:
                data = {"error": "invalid_json", "raw": resp.text[:800]}
            if resp.status_code >= 400:
                return {"error": data}
            return data

    async def rollout(
        self,
        *,
        run_id: str,
        env_name: str,
        seed: int,
        difficulty: str | None,
        policy_name: str,
        policy_config: dict[str, Any],
        max_turns: int,
        env_config: dict[str, Any] | None = None,
        ops: list[str] | None = None,
    ) -> dict[str, Any]:
        ops_seq: list[str] = list(ops) if ops is not None else []
        if not ops_seq:
            for _ in range(max_turns):
                ops_seq.extend(["agent", "env"])
        env_cfg: dict[str, Any] = {}
        if isinstance(env_config, dict):
            env_cfg.update(env_config)
        if difficulty is not None and "difficulty" not in env_cfg:
            env_cfg["difficulty"] = difficulty
        payload: dict[str, Any] = {
            "run_id": run_id,
            "env": {
                "env_name": env_name,
                "config": env_cfg,
                "seed": seed,
            },
            "policy": {
                "policy_name": policy_name,
                "config": policy_config,
            },
            "ops": ops_seq,
            "on_done": "terminate",
        }
        # Ensure X-API-Key is included
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        resp = await self.client.post("/rollout", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


TASK_APP_URL = os.getenv("TASK_APP_URL", "https://YOUR-TASK-APP.modal.run").rstrip("/")
MODEL = os.getenv("EVAL_MODEL", "qwen/qwen3-32b")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "3"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "1"))


def _interact_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Perform actions in the Crafter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["actions", "reasoning"],
                },
            },
        }
    ]


def _build_messages_from_observation(
    observation: dict[str, Any], history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    inv = observation.get("inventory") or {}
    pos = observation.get("player_position") or []
    ach = observation.get("achievements_status") or {}
    user_lines: list[str] = []
    user_lines.append("Environment: CrafterClassic")
    user_lines.append(f"Player position: {pos}")
    user_lines.append(f"Inventory: {json.dumps(inv, ensure_ascii=False)}")
    unlocked = [k for k, v in ach.items() if v]
    if unlocked:
        user_lines.append(f"Unlocked achievements: {unlocked}")
    user_lines.append("Provide 2-5 actions as a plan to explore and progress.")
    # short history summary
    if history:
        last = history[-1]
        user_lines.append(f"Last actions: {last.get('actions')}")
    content = "\n".join(user_lines)
    return [{"role": "user", "content": content}]


def _parse_tool_calls_from_openai_response(data: dict[str, Any]) -> list[str]:
    try:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                fn = tcs[0].get("function", {}) if isinstance(tcs[0], dict) else {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        obj = json.loads(args)
                    except Exception:
                        obj = {}
                elif isinstance(args, dict):
                    obj = args
                else:
                    obj = {}
                acts = obj.get("actions")
                if isinstance(acts, list):
                    return [str(a) for a in acts][:5]
    except Exception:
        pass
    # Fallback: parse JSON object from assistant text
    try:
        choices = data.get("choices")
        msg = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text"))
                for part in content
                if isinstance(part, dict) and part.get("text")
            )
        for raw in re.findall(r"\{[\s\S]*\}", text or ""):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("tool") in ("interact", None):
                    acts = obj.get("args", {}).get("actions")
                    if isinstance(acts, list):
                        return [str(a) for a in acts][:5]
            except Exception:
                continue
    except Exception:
        pass
    return []


async def _choose_actions_via_llm(
    client: TaskAppClient,
    provider: str,
    model: str,
    observation: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[str]:
    messages = _build_messages_from_observation(observation, history)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": _interact_tool_schema(),
        "max_tokens": 8192,
        "temperature": 0.2,
    }
    if provider == "groq":
        # Groq path: avoid forcing tool_choice to reduce 400 errors; proxy will synthesize if missing
        data = await client.proxy_groq_chat(payload)
    elif provider == "vllm":
        info = await client.get_info()
        vllm_base = ((info or {}).get("inference") or {}).get("base_url")
        if not vllm_base:
            return []
        # For vLLM path, we can force the single tool
        vllm_payload = dict(payload)
        vllm_payload["tool_choice"] = {"type": "function", "function": {"name": "interact"}}
        data = await client.vllm_chat(str(vllm_base), vllm_payload)
        if isinstance(data, dict) and data.get("error"):
            return []
    else:
        return []
    actions = _parse_tool_calls_from_openai_response(data)
    return actions or []


def _expand_actions_to_tool_calls(actions: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in actions[:5]:
        out.append({"tool": "interact", "args": {"action": a}})
    return out


def _detect_provider(model: str) -> str:
    m = (model or "").lower()
    if "qwen/qwen3-32b" in m or "qwen-2.5-" in m or m.startswith("groq:"):
        return "groq"
    return "vllm"


def _rollout_inference_url_from_cfg(cfg: dict[str, Any], default_vllm: str | None) -> str | None:
    # Prefer explicit inference_url in TOML; else fall back to discovered vLLM base
    url = cfg.get("inference_url")
    if isinstance(url, str) and url:
        return url
    return default_vllm


async def eval_episode(client: TaskAppClient, seed: int) -> dict[str, Any]:
    env_name = "CrafterClassic"
    history: list[dict[str, Any]] = []
    achievements: set[str] = set()
    turns = 0

    # Initialize environment
    init_cfg: dict[str, Any] = {
        "seed": seed,
        "world_config": {"difficulty": os.getenv("DIFFICULTY", "easy")},
    }
    created = await client.initialize(env_name, init_cfg)
    env_id = created.get("env_id")
    if not isinstance(env_id, str) or not env_id:
        raise RuntimeError(f"Invalid env_id from initialize: {created}")
    done = False
    provider = _detect_provider(MODEL)
    observation = created.get("observation") if isinstance(created, dict) else None
    if not isinstance(observation, dict):
        observation = {}
    try:
        ach_map_initial = observation.get("achievements_status")
        if isinstance(ach_map_initial, dict):
            achievements.update(k for k, v in ach_map_initial.items() if v)
    except Exception:
        pass

    try:
        while turns < MAX_TURNS and not done:
            # Ask LLM for actions; fallback to a simple exploratory pair
            chosen_actions = await _choose_actions_via_llm(
                client, provider, MODEL, observation, history
            )
            if not chosen_actions:
                chosen_actions = ["move_up", "do"]
            tool_calls = _expand_actions_to_tool_calls(chosen_actions)
            step = await client.step(env_name, env_id, tool_calls)
            done = bool(step.get("done"))
            turns += 1
            history.append({"actions": chosen_actions, "reasoning": "explore then interact"})
            # Update observation for next turn if available
            if isinstance(step, dict):
                nxt = step.get("observation")
                if isinstance(nxt, dict):
                    observation = nxt
                    try:
                        ach_map = observation.get("achievements_status")
                        if isinstance(ach_map, dict):
                            achievements.update(k for k, v in ach_map.items() if v)
                    except Exception:
                        pass
    finally:
        with contextlib.suppress(Exception):
            await client.terminate(env_name, env_id)

    return {"seed": seed, "turns": turns, "achievements": sorted(achievements)}


def _load_dotenv_defaults() -> None:
    """Load .env-style key/value pairs without clobbering explicit exports."""
    try:
        script_path = Path(__file__).resolve()
    except Exception:
        return
    candidates: list[Path] = []
    # Prefer the repo root .env, then allow per-directory overrides.
    for base in [Path.cwd(), script_path.parent, *script_path.parents]:
        env_path = base / ".env"
        if env_path not in candidates and env_path.is_file():
            candidates.append(env_path)
    seen: set[str] = set()
    try:
        for env_path in candidates:
            try:
                for raw in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    val = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            except Exception:
                continue
    except Exception:
        return


async def main() -> None:
    _load_dotenv_defaults()
    if not (os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY")):
        raise RuntimeError(
            "ENVIRONMENT_API_KEY is required. Export it or add it to your project .env."
        )

    parser = argparse.ArgumentParser(
        description="Baseline eval against task app with optional TOML config"
    )
    parser.add_argument("--toml", help="Path to TOML config file", default=None)
    parser.add_argument(
        "--use-rollout", action="store_true", help="Use server-side rollout endpoint for eval"
    )
    args = parser.parse_args()

    global TASK_APP_URL, MODEL, NUM_EPISODES, MAX_TURNS, CONCURRENCY
    cfg: dict[str, Any] = {}
    if args.toml:
        with open(args.toml, "rb") as f:
            cfg = tomllib.load(f)
        # Map known keys; tolerate missing
        TASK_APP_URL = (cfg.get("task_app_url") or TASK_APP_URL).rstrip("/")
        MODEL = cfg.get("model") or MODEL
        NUM_EPISODES = int(cfg.get("num_episodes") or NUM_EPISODES)
        MAX_TURNS = int(cfg.get("max_turns") or MAX_TURNS)
        CONCURRENCY = int(cfg.get("concurrency") or CONCURRENCY)
        if "difficulty" in cfg:
            os.environ["DIFFICULTY"] = str(cfg.get("difficulty"))
        # Replace placeholder URLs with env if present
        if "your-task-app.modal.run" in TASK_APP_URL.lower():
            env_url = os.getenv("TASK_APP_URL")
            if env_url:
                TASK_APP_URL = env_url.rstrip("/")
            else:
                raise RuntimeError(
                    "TASK_APP_URL is a placeholder. Set task_app_url in TOML or export TASK_APP_URL."
                )

    print(f"Task App: {TASK_APP_URL}")
    print(
        f"Model: {MODEL} Episodes: {NUM_EPISODES} Max turns: {MAX_TURNS} Concurrency: {CONCURRENCY}"
    )
    sem = asyncio.Semaphore(max(CONCURRENCY, 1))
    async with TaskAppClient(TASK_APP_URL, api_key=os.getenv("ENVIRONMENT_API_KEY")) as client:
        if args.use_rollout:
            # Use server-side rollout; derive inference URL per provider
            info = await client.get_info()
            default_vllm = ((info or {}).get("inference") or {}).get("base_url")
            inf_url = _rollout_inference_url_from_cfg(cfg, default_vllm)
            if not inf_url:
                raise RuntimeError("Could not resolve inference URL for rollout")

            async def _run(seed: int):
                async with sem:
                    try:
                        run_id = f"eval-{seed}"
                        rollout_cfg_raw = cfg.get("rollout") or {}
                        rollout_cfg = (
                            dict(rollout_cfg_raw) if isinstance(rollout_cfg_raw, dict) else {}
                        )
                        env_config_raw = rollout_cfg.get("env_config") or {}
                        env_config = (
                            deepcopy(env_config_raw) if isinstance(env_config_raw, dict) else {}
                        )
                        policy_cfg_raw = rollout_cfg.get("policy_config") or {}
                        policy_cfg = (
                            deepcopy(policy_cfg_raw) if isinstance(policy_cfg_raw, dict) else {}
                        )
                        policy_cfg.setdefault("model", cfg.get("model", MODEL))
                        policy_cfg.setdefault("inference_url", inf_url)
                        for k in (
                            "max_tokens",
                            "temperature",
                            "top_p",
                            "thinking_mode",
                            "thinking_budget",
                            "use_tools",
                        ):
                            if k in cfg and cfg.get(k) is not None and k not in policy_cfg:
                                policy_cfg[k] = cfg.get(k)

                        env_name = str(rollout_cfg.get("env_name") or "crafter")
                        policy_name = str(
                            rollout_cfg.get("policy_name") or cfg.get("policy_name") or "crafter"
                        )

                        max_turns_local = MAX_TURNS
                        for candidate in (rollout_cfg.get("max_turns"), cfg.get("max_turns")):
                            if candidate is None:
                                continue
                            with contextlib.suppress(Exception):
                                max_turns_local = int(candidate)
                                break

                        difficulty_override: str | None = None
                        if isinstance(env_config, dict):
                            diff_cfg = env_config.get("difficulty")
                            if isinstance(diff_cfg, str) and diff_cfg:
                                difficulty_override = diff_cfg
                        if difficulty_override is None:
                            cfg_diff = rollout_cfg.get("difficulty") or cfg.get("difficulty")
                            if isinstance(cfg_diff, str) and cfg_diff:
                                difficulty_override = cfg_diff
                        if difficulty_override is None:
                            difficulty_override = os.getenv("DIFFICULTY", "easy")

                        r = await client.rollout(
                            run_id=run_id,
                            env_name=env_name,
                            seed=seed,
                            difficulty=difficulty_override,
                            policy_name=policy_name,
                            policy_config=policy_cfg,
                            max_turns=max_turns_local,
                            env_config=env_config,
                        )
                        metrics_block = r.get("metrics") or {}
                        mean_return = None
                        if isinstance(metrics_block, dict):
                            with contextlib.suppress(Exception):
                                mean_return = float(metrics_block.get("mean_return"))
                        stepwise_details: dict[str, Any] = {}
                        if isinstance(metrics_block, dict):
                            details_block = metrics_block.get("details") or {}
                            if isinstance(details_block, dict):
                                step_block = details_block.get("stepwise") or {}
                                if isinstance(step_block, dict):
                                    stepwise_details = step_block
                        # Extract achievements count if present
                        achieved: set[str] = set()
                        try:
                            trajs = r.get("trajectories") or []
                            final_obs = (
                                (trajs[0].get("final") or {}).get("observation")
                                if trajs and isinstance(trajs[0], dict)
                                else None
                            )
                            ach_map = (
                                (final_obs or {}).get("achievements_status")
                                if isinstance(final_obs, dict)
                                else None
                            )
                            if isinstance(ach_map, dict):
                                achieved.update(k for k, v in ach_map.items() if v)
                        except Exception:
                            pass
                        try:
                            step_seen = stepwise_details.get("unique_achievements")
                        except Exception:
                            step_seen = None
                        if isinstance(step_seen, (list, tuple, set)):
                            achieved.update(str(a) for a in step_seen)
                        else:
                            try:
                                alt_seen = stepwise_details.get("achievements_seen")
                                if isinstance(alt_seen, (list, tuple, set)):
                                    achieved.update(str(a) for a in alt_seen)
                            except Exception:
                                pass
                        try:
                            summary_final = stepwise_details.get("final_achievements")
                            if isinstance(summary_final, (list, tuple, set)):
                                achieved.update(str(a) for a in summary_final)
                        except Exception:
                            pass
                        ach = sorted(achieved)
                        length = 0
                        try:
                            trajs = r.get("trajectories") or []
                            if trajs and isinstance(trajs[0], dict):
                                length = int(trajs[0].get("length") or 0)
                        except Exception:
                            pass
                        return {
                            "seed": seed,
                            "turns": length,
                            "achievements": ach,
                            "mean_return": mean_return,
                            "stepwise": stepwise_details,
                        }
                    except Exception as e:
                        return {
                            "seed": seed,
                            "turns": 0,
                            "achievements": [],
                            "mean_return": None,
                            "stepwise": {},
                            "error": str(e),
                        }

            results = await asyncio.gather(
                *[asyncio.create_task(_run(i)) for i in range(1, NUM_EPISODES + 1)],
                return_exceptions=False,
            )
            # Aggregate summary
            counts = [len(r.get("achievements") or []) for r in results if isinstance(r, dict)]
            turns = [int(r.get("turns") or 0) for r in results if isinstance(r, dict)]
            all_ach = Counter()
            for r in results:
                try:
                    for a in r.get("achievements") or []:
                        all_ach[a] += 1
                except Exception:
                    pass
            mean_returns: list[float] = []
            stepwise_reward_sums: list[float] = []
            stepwise_indicator_sums: list[float] = []
            stepwise_new_ach_totals: list[float] = []
            stepwise_resource_rewards: list[float] = []
            strategies_seen = Counter()
            unique_union: set[str] = set()
            final_union: set[str] = set()
            for r in results:
                if not isinstance(r, dict):
                    continue
                with contextlib.suppress(Exception):
                    mean_val = r.get("mean_return")
                    if mean_val is not None:
                        mean_returns.append(float(mean_val))
                stepwise_block = r.get("stepwise")
                if isinstance(stepwise_block, dict) and stepwise_block:
                    with contextlib.suppress(Exception):
                        if stepwise_block.get("reward_sum") is not None:
                            stepwise_reward_sums.append(float(stepwise_block.get("reward_sum")))
                    with contextlib.suppress(Exception):
                        if stepwise_block.get("indicator_sum") is not None:
                            stepwise_indicator_sums.append(float(stepwise_block.get("indicator_sum")))
                    with contextlib.suppress(Exception):
                        if stepwise_block.get("new_achievements_total") is not None:
                            stepwise_new_ach_totals.append(
                                float(stepwise_block.get("new_achievements_total"))
                            )
                    with contextlib.suppress(Exception):
                        if stepwise_block.get("resource_reward") is not None:
                            stepwise_resource_rewards.append(
                                float(stepwise_block.get("resource_reward"))
                            )
                    with contextlib.suppress(Exception):
                        uniq = stepwise_block.get("unique_achievements") or []
                        if isinstance(uniq, (list, tuple, set)):
                            unique_union.update(str(v) for v in uniq)
                    with contextlib.suppress(Exception):
                        final = stepwise_block.get("final_achievements") or []
                        if isinstance(final, (list, tuple, set)):
                            final_union.update(str(v) for v in final)
                    strategy_name = stepwise_block.get("strategy")
                    if isinstance(strategy_name, str) and strategy_name:
                        strategies_seen[strategy_name] += 1
            aggregate: dict[str, Any] = {
                "completed": sum(
                    1 for r in results if isinstance(r, dict) and not r.get("error")
                ),
                "total": len(results),
                "avg_turns": (sum(turns) / len(turns)) if turns else 0.0,
                "avg_achievements": (sum(counts) / len(counts)) if counts else 0.0,
                "achievements_freq": dict(all_ach),
            }
            if mean_returns:
                aggregate["avg_mean_return"] = sum(mean_returns) / len(mean_returns)
            if stepwise_reward_sums:
                aggregate["avg_stepwise_reward_sum"] = sum(stepwise_reward_sums) / len(
                    stepwise_reward_sums
                )
            if stepwise_indicator_sums:
                aggregate["avg_stepwise_indicator_sum"] = sum(stepwise_indicator_sums) / len(
                    stepwise_indicator_sums
                )
            if stepwise_new_ach_totals:
                aggregate["avg_stepwise_new_achievements"] = sum(stepwise_new_ach_totals) / len(
                    stepwise_new_ach_totals
                )
            if stepwise_resource_rewards:
                aggregate["avg_stepwise_resource_reward"] = (
                    sum(stepwise_resource_rewards) / len(stepwise_resource_rewards)
                )
            if strategies_seen:
                aggregate["stepwise_strategies"] = dict(strategies_seen)
            aggregate["stepwise_samples"] = max(
                len(stepwise_reward_sums),
                len(stepwise_indicator_sums),
                len(stepwise_new_ach_totals),
                len(stepwise_resource_rewards),
            ) if any(
                (
                    stepwise_reward_sums,
                    stepwise_indicator_sums,
                    stepwise_new_ach_totals,
                    stepwise_resource_rewards,
                )
            ) else 0
            if not unique_union:
                for r in results:
                    try:
                        for a in r.get("achievements") or []:
                            unique_union.add(str(a))
                    except Exception:
                        continue
            if not final_union:
                final_union.update(unique_union)
            if unique_union:
                aggregate["unique_achievements_union"] = sorted(unique_union)
            if final_union:
                aggregate["final_achievements_union"] = sorted(final_union)
            summary = {
                "episodes": results,
                "aggregate": aggregate,
            }
            print(json.dumps(summary, indent=2))
            # Failure guardrails: any error or zero-turn episodes across the board
            any_errors = any(isinstance(r, dict) and r.get("error") for r in results)
            all_zero_turns = all((int(r.get("turns") or 0) == 0) for r in results if isinstance(r, dict))
            if any_errors or all_zero_turns:
                # Exit non-zero so automation/CI treats this as a failure
                sys.exit(2)
        else:

            async def _run(seed: int):
                async with sem:
                    return await eval_episode(client, seed)

            results = await asyncio.gather(
                *[asyncio.create_task(_run(i)) for i in range(1, NUM_EPISODES + 1)]
            )
            print(json.dumps({"episodes": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
