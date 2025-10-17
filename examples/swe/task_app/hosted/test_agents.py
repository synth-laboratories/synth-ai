#!/usr/bin/env python3
"""
Smoke test for Wordle and Sokoban ReAct agents using the hosted service.

Prereqs:
- Run the service: python examples/swe/task_app/hosted/main.py
- Run an OpenAI-compatible inference server (e.g., Flash/vLLM) at VLLM_BASE_URL
  that serves model "gpt-5-nano" or adjust MODEL below.

This script will:
- Create a Wordle/Sokoban env
- Create corresponding *-react policy with tools
- Ask the policy for tool_calls via /policy/step (which calls the model)
- Apply tool_calls to the env via /env/step
"""

import asyncio
import os

import httpx

BASE_URL = os.environ.get("SYNTH_ENVS_HOSTED_URL", "http://localhost:8000")
INFER_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
MODEL = os.environ.get("MODEL", "gpt-5-nano")


async def run_wordle(rounds: int = 3) -> None:
    async with httpx.AsyncClient() as client:
        # Create env
        resp = await client.post(
            f"{BASE_URL}/env/create",
            json={
                "env_name": "Wordle",
                "config": {"word_length": 5, "max_guesses": 6},
                "seed": 0,
                "rl_run_id": "agents-smoke",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        env_id = data["env_id"]
        obs = data["observation"]
        print("Wordle env created:", env_id)

        # Create policy
        resp = await client.post(
            f"{BASE_URL}/policy/create",
            json={
                "policy_name": "wordle-react",
                "config": {
                    "inference_url": INFER_URL,
                    "model": MODEL,
                    "use_tools": True,
                    "word_length": 5,
                    "max_guesses": 6,
                },
                "rl_run_id": "agents-smoke",
                "bound_env_id": env_id,
            },
        )
        resp.raise_for_status()
        policy_id = resp.json()["policy_id"]
        print("Wordle policy:", policy_id)

        # Loop a few rounds
        for i in range(rounds):
            print(f"[Wordle] Round {i + 1}")
            step_req = {"policy_id": policy_id, "observation": obs, "dry_run": False}
            resp = await client.post(f"{BASE_URL}/policy/step", json=step_req)
            resp.raise_for_status()
            step_out = resp.json()
            tool_calls = step_out.get("tool_calls", [])
            print(" tool_calls:", tool_calls)
            if not tool_calls:
                break
            resp = await client.post(
                f"{BASE_URL}/env/step",
                json={"env_id": env_id, "tool_calls": tool_calls},
            )
            resp.raise_for_status()
            env_step = resp.json()
            obs = env_step["observation"]
            print(" done:", env_step.get("done"), "reward:", env_step.get("reward"))
            if env_step.get("done"):
                break


async def run_sokoban(rounds: int = 3) -> None:
    async with httpx.AsyncClient() as client:
        # Create env (no initial_state provided; relies on env default)
        resp = await client.post(
            f"{BASE_URL}/env/create",
            json={
                "env_name": "Sokoban",
                "config": {"difficulty": "easy"},
                "seed": 0,
                "rl_run_id": "agents-smoke",
            },
        )
        if resp.status_code != 200:
            print("Sokoban create failed:", resp.status_code, resp.text)
            return
        data = resp.json()
        env_id = data["env_id"]
        obs = data["observation"]
        print("Sokoban env created:", env_id)

        resp = await client.post(
            f"{BASE_URL}/policy/create",
            json={
                "policy_name": "sokoban-react",
                "config": {
                    "inference_url": INFER_URL,
                    "model": MODEL,
                    "use_tools": True,
                },
                "rl_run_id": "agents-smoke",
                "bound_env_id": env_id,
            },
        )
        if resp.status_code != 200:
            print("Sokoban policy create failed:", resp.status_code, resp.text)
            return
        policy_id = resp.json()["policy_id"]
        print("Sokoban policy:", policy_id)

        for i in range(rounds):
            print(f"[Sokoban] Round {i + 1}")
            step_req = {"policy_id": policy_id, "observation": obs, "dry_run": False}
            resp = await client.post(f"{BASE_URL}/policy/step", json=step_req)
            if resp.status_code != 200:
                print(" policy step failed:", resp.status_code, resp.text)
                break
            step_out = resp.json()
            tool_calls = step_out.get("tool_calls", [])
            print(" tool_calls:", tool_calls)
            if not tool_calls:
                break
            resp = await client.post(
                f"{BASE_URL}/env/step",
                json={"env_id": env_id, "tool_calls": tool_calls},
            )
            if resp.status_code != 200:
                print(" env step failed:", resp.status_code, resp.text)
                break
            env_step = resp.json()
            obs = env_step["observation"]
            print(" done:", env_step.get("done"), "reward:", env_step.get("reward"))
            if env_step.get("done"):
                break


async def main():
    print("Testing Wordle agent with model:", MODEL)
    await run_wordle(rounds=3)
    print("\nTesting Sokoban agent with model:", MODEL)
    await run_sokoban(rounds=3)


if __name__ == "__main__":
    asyncio.run(main())
