"""Create an Archipelago environment pool for the demo."""

from __future__ import annotations

import argparse
import os

import httpx


def _env_str(name: str, fallback: str = "") -> str:
    return (os.environ.get(name) or fallback).strip()


def _env_int(name: str, fallback: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _archipelago_config() -> dict[str, object]:
    return {
        "env_image": _env_str("RHODES_APEX_ENV_IMAGE"),
        "agent_image": _env_str("RHODES_APEX_AGENT_IMAGE"),
        "grading_image": _env_str("RHODES_APEX_GRADING_IMAGE"),
        "env_port": _env_int("ARCHIPELAGO_ENV_PORT"),
        "initial_snapshot_path": _env_str(
            "ARCHIPELAGO_INITIAL_SNAPSHOT_PATH", "/configs/original_snapshot.zip"
        ),
        "mcp_config_path": _env_str("ARCHIPELAGO_MCP_CONFIG_PATH", "/configs/mcp_config.json"),
        "initial_messages_path": _env_str(
            "ARCHIPELAGO_INITIAL_MESSAGES_PATH", "/configs/initial_messages.json"
        ),
        "agent_config_path": _env_str("ARCHIPELAGO_AGENT_CONFIG_PATH", "/configs/agent_config.json"),
        "orchestrator_config_path": _env_str(
            "ARCHIPELAGO_ORCHESTRATOR_CONFIG_PATH", "/configs/orchestrator_config.json"
        ),
        "grading_settings_path": _env_str(
            "ARCHIPELAGO_GRADING_SETTINGS_PATH", "/configs/grading_settings.json"
        ),
        "verifiers_path": _env_str("ARCHIPELAGO_VERIFIERS_PATH", "/configs/verifiers.json"),
        "eval_configs_path": _env_str("ARCHIPELAGO_EVAL_CONFIGS_PATH", "/configs/eval_configs.json"),
        "scoring_config_path": _env_str(
            "ARCHIPELAGO_SCORING_CONFIG_PATH", "/configs/scoring_config.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Archipelago environment pool")
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("SYNTH_BACKEND_URL")
        or os.environ.get("SYNTH_BASE_URL")
        or "https://api-dev.usesynth.ai",
    )
    parser.add_argument("--pool-id", default=None, help="Optional pool_id override")
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--policy-tag", action="append", default=["archipelago"])
    args = parser.parse_args()

    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY is required")

    archipelago = _archipelago_config()
    missing = [k for k in ("env_image", "agent_image", "grading_image") if not archipelago.get(k)]
    if missing:
        raise SystemExit(
            "Missing required Archipelago image(s): "
            + ", ".join(missing)
            + " (set RHODES_APEX_*_IMAGE env vars)"
        )

    request = {
        "pool_id": args.pool_id,
        "pool_type": "archipelago",
        "capacity": args.capacity,
        "concurrency": args.concurrency,
        "policy_tags": args.policy_tag,
        "tasks": [
            {
                "task_id": "archipelago-simple-task",
                "backend": "archipelago",
                "archipelago": archipelago,
            }
        ],
    }

    backend_base = args.backend_url.rstrip("/")
    use_infra = "infra-api" in backend_base or os.environ.get("ENV_POOLS_USE_INFRA_API") in (
        "1",
        "true",
        "yes",
        "on",
    )
    if use_infra:
        url = f"{backend_base}/v1/pools"
        headers = {"x-user-api-key": api_key}
    else:
        url = f"{backend_base}/api/v1/environment-pools/pools"
        headers = {"Authorization": f"Bearer {api_key}"}

    resp = httpx.post(url, headers=headers, json=request, timeout=30.0)
    resp.raise_for_status()
    pool = resp.json()
    pool_id = pool.get("pool", {}).get("pool_id") if isinstance(pool, dict) else None
    print(f"Created pool: {pool_id}")


if __name__ == "__main__":
    main()
