#!/usr/bin/env python3
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import httpx

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
LOG_PATH = ROOT / "nohup_sokoban_eval.log"
BASE_URL = "http://127.0.0.1:8911"
ROLLOUTS = 5
TIMEOUT_S = 45.0


def load_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        raise SystemExit(f"Missing {ENV_PATH}")
    env = os.environ.copy()
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env[key] = value
    for required in ("ENVIRONMENT_API_KEY", "GROQ_API_KEY"):
        if not env.get(required):
            raise SystemExit(f"{required} missing in .env")
    return env


async def wait_for_health(client: httpx.AsyncClient, headers: dict[str, str]) -> None:
    for _ in range(60):
        try:
            resp = await client.get(f"{BASE_URL}/health", headers=headers, timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    raise RuntimeError("Task app never became healthy")


async def run_rollout(client: httpx.AsyncClient, run_id: str, seed: int) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "env": {
            "env_name": "sokoban",
            "seed": seed,
            "config": {"difficulty": "easy", "max_steps": 50},
        },
        "policy": {
            "policy_name": "sokoban-react",
            "config": {
                "provider": "groq",
                "model": "qwen/qwen3-32b",
                "temperature": 0.0,
                "max_tokens": 128,
            },
        },
        "ops": ["agent", "env"],
        "on_done": "terminate",
    }
    headers = {"x-api-key": client.headers["x-api-key"], "content-type": "application/json"}
    try:
        resp = await client.post(
            f"{BASE_URL}/rollout",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        traj = data["trajectories"][0]
        final_obs = traj["final"]["observation"]
        reward = float(traj["final"].get("reward") or 0.0)
        solved = final_obs.get("boxes_on_target") == final_obs.get("num_boxes")
        return {
            "run_id": run_id,
            "reward": reward,
            "solved": bool(solved),
            "steps": traj.get("length"),
        }
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        return {"run_id": run_id, "error": f"{exc!r}", "response": detail}
    except Exception as exc:  # surface error details for debugging
        return {"run_id": run_id, "error": repr(exc)}


async def main() -> None:
    env = load_env()
    log_file = open(LOG_PATH, "w")
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "-m",
            "synth_ai",
            "task-app",
            "serve",
            "sokoban",
            "--port",
            "8911",
            "--no-reload",
            "--trace",
            "./traces/v3",
            "--trace-db",
            "./traces/v3/synth_ai.db",
        ],
        stdout=log_file,
        stderr=log_file,
        env=env,
    )

    try:
        async with httpx.AsyncClient() as client:
            client.headers["x-api-key"] = env["ENVIRONMENT_API_KEY"]
            await wait_for_health(client, {"x-api-key": env["ENVIRONMENT_API_KEY"]})
            tasks = [
                run_rollout(client, f"sokoban-groq-eval-{i}", 100 + i)
                for i in range(ROLLOUTS)
            ]
            results = await asyncio.gather(*tasks)
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        log_file.close()

    rewards = [r["reward"] for r in results if "reward" in r]
    solved = [r["solved"] for r in results if "solved" in r]
    summary = {
        "rollouts": results,
        "average_reward": mean(rewards) if rewards else 0.0,
        "solved_rate": sum(solved) / len(solved) if solved else 0.0,
    }
    output_path = ROOT / "scripts" / "run_groq_eval_output.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
