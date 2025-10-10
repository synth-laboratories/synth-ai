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
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

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
        difficulty: str,
        policy_name: str,
        policy_config: dict[str, Any],
        max_turns: int,
    ) -> dict[str, Any]:
        ops: list[str] = []
        for _ in range(max_turns):
            ops.extend(["agent", "env"])
        payload: dict[str, Any] = {
            "run_id": run_id,
            "env": {
                "env_name": env_name,
                "config": {"difficulty": difficulty},
                "seed": seed,
            },
            "policy": {
                "policy_name": policy_name,
                "config": policy_config,
            },
            "ops": ops,
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
    finally:
        with contextlib.suppress(Exception):
            await client.terminate(env_name, env_id)

    return {"seed": seed, "turns": turns, "achievements": sorted(achievements)}


async def main() -> None:
    # Best-effort load local .env if present (ensures ENVIRONMENT_API_KEY for rollout)
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass

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
                        # Build policy config from TOML (explicit control; no server-side guessing)
                        policy_cfg: dict[str, Any] = {
                            "model": cfg.get("model", MODEL),
                            "inference_url": inf_url,
                        }
                        for k in (
                            "max_tokens",
                            "temperature",
                            "top_p",
                            "thinking_mode",
                            "thinking_budget",
                            "use_tools",
                        ):
                            if k in cfg and cfg.get(k) is not None:
                                policy_cfg[k] = cfg.get(k)

                        r = await client.rollout(
                            run_id=run_id,
                            env_name="crafter",
                            seed=seed,
                            difficulty=os.getenv("DIFFICULTY", "easy"),
                            policy_name=cfg.get("policy_name", "crafter"),
                            policy_config=policy_cfg,
                            max_turns=MAX_TURNS,
                        )
                        # Extract achievements count if present
                        ach = []
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
                                ach = sorted([k for k, v in ach_map.items() if v])
                        except Exception:
                            pass
                        length = 0
                        try:
                            trajs = r.get("trajectories") or []
                            if trajs and isinstance(trajs[0], dict):
                                length = int(trajs[0].get("length") or 0)
                        except Exception:
                            pass
                        return {"seed": seed, "turns": length, "achievements": ach}
                    except Exception as e:
                        return {"seed": seed, "turns": 0, "achievements": [], "error": str(e)}

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
            summary = {
                "episodes": results,
                "aggregate": {
                    "completed": sum(1 for r in results if not r.get("error")),
                    "total": len(results),
                    "avg_turns": (sum(turns) / len(turns)) if turns else 0.0,
                    "avg_achievements": (sum(counts) / len(counts)) if counts else 0.0,
                    "achievements_freq": dict(all_ach),
                },
            }
            print(json.dumps(summary, indent=2))
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
