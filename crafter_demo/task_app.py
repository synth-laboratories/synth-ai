
"""Demo-friendly wrapper for the GRPO Crafter task app."""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_ai.task.apps import ModalDeploymentConfig, registry
from synth_ai.task.apps.grpo_crafter import build_config
from synth_ai.task.server import TaskAppConfig, create_task_app, run_task_app


APP_ID = "grpo-crafter-demo"
BASE_APP_ID = "grpo-crafter"


_BASE_CONFIG = build_config()
TASK_APP_CONFIG = TaskAppConfig(
    app_id="grpo-crafter-demo",
    name=_BASE_CONFIG.name,
    description=_BASE_CONFIG.description,
    base_task_info=_BASE_CONFIG.base_task_info,
    describe_taskset=_BASE_CONFIG.describe_taskset,
    provide_task_instances=_BASE_CONFIG.provide_task_instances,
    rollout=_BASE_CONFIG.rollout,
    dataset_registry=_BASE_CONFIG.dataset_registry,
    rubrics=_BASE_CONFIG.rubrics,
    proxy=_BASE_CONFIG.proxy,
    routers=_BASE_CONFIG.routers,
    middleware=_BASE_CONFIG.middleware,
    app_state=_BASE_CONFIG.app_state,
    require_api_key=_BASE_CONFIG.require_api_key,
    expose_debug_env=_BASE_CONFIG.expose_debug_env,
    cors_origins=_BASE_CONFIG.cors_origins,
    startup_hooks=_BASE_CONFIG.startup_hooks,
    shutdown_hooks=_BASE_CONFIG.shutdown_hooks,
)

try:
    _BASE_ENTRY = registry.get(BASE_APP_ID)
except Exception:  # pragma: no cover - registry may be unavailable
    MODAL_DEPLOYMENT: ModalDeploymentConfig | None = None
else:
    base_modal = _BASE_ENTRY.modal
    if base_modal is None:
        MODAL_DEPLOYMENT = None
    else:
        modal_app_name = base_modal.app_name
        if not modal_app_name.endswith("-demo"):
            modal_app_name = f"{modal_app_name}-demo"
        MODAL_DEPLOYMENT = ModalDeploymentConfig(
            app_name=modal_app_name,
            python_version=base_modal.python_version,
            pip_packages=tuple(base_modal.pip_packages),
            extra_local_dirs=tuple(base_modal.extra_local_dirs),
            secret_names=tuple(base_modal.secret_names),
            volume_mounts=tuple(base_modal.volume_mounts),
            timeout=base_modal.timeout,
            memory=base_modal.memory,
            cpu=base_modal.cpu,
            min_containers=base_modal.min_containers,
            max_containers=base_modal.max_containers,
        )

ENV_FILES: tuple[str, ...] = ()


def build_task_app_config() -> TaskAppConfig:
    """Return a fresh TaskAppConfig for the demo wrapper."""

    return TASK_APP_CONFIG.clone()


def fastapi_app():
    """Return the FastAPI application for Modal or other ASGI hosts."""

    return create_task_app(build_task_app_config())


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
        build_task_app_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )
