"""Legacy entrypoint for the math single-step task app."""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_ai.task.server import create_task_app, run_task_app
from synth_ai.task.apps.math_single_step import build_config


def fastapi_app():
    """Return a FastAPI application for hosting the math task app."""

    return create_task_app(build_config())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the math single-step task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[2] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )

