"""Compatibility wrapper for the GRPO Crafter task app.

This module now delegates to the shared TaskAppConfig defined in
`synth_ai.task.apps.grpo_crafter`. It is kept for legacy usage (running the
file directly or targeting `fastapi_app` from external tooling). Prefer using
`uvx synth-ai serve grpo-crafter` for local development and testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_ai.task.server import create_task_app, run_task_app
from synth_ai.task.apps.grpo_crafter import build_config


def fastapi_app():
    """Return the FastAPI application for Modal or other ASGI hosts."""

    return create_task_app(build_config())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Crafter task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[4] / "backend" / ".env.dev"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )
