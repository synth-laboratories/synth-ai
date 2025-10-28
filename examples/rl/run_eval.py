#!/usr/bin/env python3
"""Evaluate math single-step task policies against the task app."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import tomllib
from typing import Any

import httpx


class TaskAppClient:
    """Minimal async client for math single-step task app."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TaskAppClient:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(120.0),
                follow_redirects=True,
            )
        return self._client

    async def initialize(self, split: str, seed: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"config": {"split": split}}
        if seed is not None:
            payload["seed"] = seed
        resp = await self.client.post("/env/math/initialize", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def step(self, env_id: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {"env_id": env_id, "action": {"tool_calls": tool_calls}}
        resp = await self.client.post("/env/math/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def terminate(self, env_id: str) -> None:
        with contextlib.suppress(Exception):
            await self.client.post("/env/math/terminate", json={"env_id": env_id})

    async def get_info(self) -> dict[str, Any]:
        resp = await self.client.get("/info")
        resp.raise_for_status()
        return resp.json()

    async def rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/rollout", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def post_inference(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as c:
            resp = await c.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


TOOL_NAME = "math_submit"
DEFAULT_SPLIT = os.getenv("MATH_EVAL_DEFAULT_SPLIT", "validation")


def _math_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Submit the final answer for the math problem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Final answer in simplest form",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Optional explanation of reasoning",
                        },
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _build_messages(problem: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve math problems. Always respond with a single math_submit tool call "
                "containing only the final answer."
            ),
        },
        {
            "role": "user",
            "content": f"Problem:\n{problem}\nReturn the final answer via math_submit.",
        },
    ]


def _parse_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    raw_calls = message.get("tool_calls") or []
    tool_calls: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        parsed_args: dict[str, Any]
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except Exception:
                parsed_args = {}
        elif isinstance(arguments, dict):
            parsed_args = dict(arguments)
        else:
            parsed_args = {}
        tool_calls.append({"tool": name, "args": parsed_args})
    return tool_calls


def _detect_provider(model: str, hint: str | None) -> str:
    if hint:
        return hint.lower()
    lowered = (model or "").lower()
    if lowered.startswith("groq:"):
        return "groq"
    return "generic"


def _resolve_inference_url(base_url: str) -> str:
    normalized = (base_url or "").rstrip("/")
    if not normalized:
        raise RuntimeError("inference_url cannot be empty")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    if "/v1/" in normalized:
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


async def _choose_actions(
    client: TaskAppClient,
    provider: str,
    model: str,
    problem: str,
    policy_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages = _build_messages(problem)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": _math_tool_schema(),
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "temperature": policy_cfg.get("temperature", 0.0),
        "top_p": policy_cfg.get("top_p", 1.0),
        "max_tokens": policy_cfg.get("max_tokens", 256),
    }

    if provider == "groq":
        # Task app proxies Groq requests; reuse existing headers on the client
        response = await client.client.post("/proxy/groq/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
    else:
        inference_url = policy_cfg.get("inference_url")
        if not inference_url:
            raise RuntimeError("inference_url required for non-groq evaluations")
        headers = dict(policy_cfg.get("headers") or {})
        for key, value in (policy_cfg.get("extra_headers") or {}).items():
            headers.setdefault(key, value)
        final_url = _resolve_inference_url(inference_url)
        try:
            response = await client.client.post(
                final_url,
                json=payload,
                headers=headers or None,
            )
        except httpx.ReadTimeout as exc:
            raise RuntimeError("Inference request timed out. Check the inference service.") from exc
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text[:800]}
        if response.status_code >= 500:
            raise RuntimeError(f"Inference server error {response.status_code}: {body}")
        if response.status_code >= 400:
            raise RuntimeError(f"Inference request invalid ({response.status_code}): {body}")
    tool_calls = _parse_tool_calls(body)
    return tool_calls, body


def _tool_to_answer(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return ""
    args = tool_calls[0].get("args") or {}
    answer = str(args.get("answer") or "")
    return answer.strip()


async def eval_episode(
    client: TaskAppClient,
    *,
    split: str,
    seed: int | None,
    model: str,
    provider: str,
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    created = await client.initialize(split, seed)
    env_id = created["env_id"]
    observation = created.get("observation") or {}
    problem = observation.get("problem") or ""

    tool_calls, raw_response = await _choose_actions(client, provider, model, problem, policy_cfg)
    answer = _tool_to_answer(tool_calls)
    result = await client.step(env_id, tool_calls)
    await client.terminate(env_id)

    info = result.get("info") or {}
    reward = result.get("reward") or 0.0
    status = info.get("status") or ("correct" if reward > 0 else "incorrect")
    return {
        "seed": seed,
        "split": split,
        "problem": problem,
        "answer": answer,
        "expected": info.get("expected_answer"),
        "reward": reward,
        "status": status,
        "correct": bool(info.get("correct")),
        "raw_response": raw_response,
        "tool_calls": tool_calls,
    }


async def eval_via_rollout(
    client: TaskAppClient,
    *,
    run_id: str,
    split: str,
    seed: int | None,
    model: str,
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "env": {
            "env_name": "math",
            "config": {"split": split},
            "seed": seed,
        },
        "policy": {
            "policy_name": "math-single-step",
            "config": policy_cfg,
        },
        "ops": ["agent", "env"],
        "on_done": "terminate",
    }
    resp = await client.rollout(payload)
    trajs = resp.get("trajectories") or []
    if not trajs:
        return {"reward": 0.0, "correct": False, "status": "missing"}
    traj = trajs[0]
    steps = traj.get("steps") or []
    step = steps[0] if steps else {}
    info = step.get("info") or {}
    observation = step.get("obs") or {}
    return {
        "seed": seed,
        "split": split,
        "problem": observation.get("problem"),
        "answer": _tool_to_answer(step.get("tool_calls") or []),
        "expected": info.get("expected_answer"),
        "reward": step.get("reward") or 0.0,
        "status": info.get("status"),
        "correct": bool(info.get("correct")),
        "raw_response": resp,
        "tool_calls": step.get("tool_calls") or [],
    }


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _default_policy_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    policy = dict(cfg.get("policy") or {})
    if "inference_url" not in policy:
        env_url = os.getenv("INFERENCE_URL")
        if env_url:
            policy["inference_url"] = env_url
    for key in ("max_tokens", "temperature", "top_p", "headers", "extra_headers"):
        if key not in policy and key in cfg:
            policy[key] = cfg[key]
    extra_headers = dict(policy.get("extra_headers") or {})
    headers = dict(policy.get("headers") or {})
    if "Authorization" not in headers and "Authorization" not in extra_headers:
        synth_key = os.getenv("SYNTH_API_KEY")
        if synth_key:
            extra_headers["Authorization"] = f"Bearer {synth_key}"
    if extra_headers:
        policy["extra_headers"] = extra_headers
    return policy


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate math task app policies")
    parser.add_argument("--toml", help="Path to TOML config", default=None)
    parser.add_argument("--use-rollout", action="store_true", help="Use server-side rollout")
    args = parser.parse_args()

    cfg = _load_config(args.toml)
    task_app_url = (cfg.get("task_app_url") or os.getenv("TASK_APP_URL") or "").rstrip("/")
    if not task_app_url:
        raise RuntimeError("task_app_url missing; set in TOML or export TASK_APP_URL")
    model = cfg.get("model") or os.getenv("EVAL_MODEL") or "groq:qwen-2.5-7b"
    split = cfg.get("split") or os.getenv("EVAL_SPLIT") or DEFAULT_SPLIT
    episodes = int(cfg.get("num_episodes") or os.getenv("NUM_EPISODES") or 50)
    seed_start = int(cfg.get("seed_start") or 0)

    policy_cfg = _default_policy_cfg(cfg)
    provider_hint = (
        cfg.get("provider") or cfg.get("policy", {}).get("provider") or policy_cfg.get("provider")
    )
    provider = _detect_provider(model, provider_hint)
    policy_cfg.pop("provider", None)

    api_key = os.getenv("ENVIRONMENT_API_KEY")

    successes = 0
    failures: dict[str, int] = {}
    results: list[dict[str, Any]] = []

    async with TaskAppClient(task_app_url, api_key=api_key) as client:
        for episode in range(episodes):
            seed = seed_start + episode
            if args.use_rollout:
                data = await eval_via_rollout(
                    client,
                    run_id=f"eval-{seed}",
                    split=split,
                    seed=seed,
                    model=model,
                    policy_cfg={"model": model, **policy_cfg},
                )
            else:
                data = await eval_episode(
                    client,
                    split=split,
                    seed=seed,
                    model=model,
                    provider=provider,
                    policy_cfg={"model": model, **policy_cfg},
                )
            results.append(data)
            if data.get("correct"):
                successes += 1
            status = data.get("status") or "unknown"
            failures[status] = failures.get(status, 0) + (0 if data.get("correct") else 1)
            answer = data.get("answer")
            expected = data.get("expected")
            problem = data.get("problem")
            tool_calls = data.get("tool_calls") or []
            print(
                f"Episode {episode + 1}/{episodes} seed={seed} status={status} reward={data.get('reward')}\n"
                f"  problem: {problem!r}\n"
                f"  tool   : {tool_calls!r}\n"
                f"  answer : {answer!r}\n  expected: {expected!r}",
                flush=True,
            )

    accuracy = successes / max(episodes, 1)
    print("=== Evaluation Summary ===")
    print(f"Task App: {task_app_url}")
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"Episodes: {episodes}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Failure breakdown:")
    for status, count in sorted(failures.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
