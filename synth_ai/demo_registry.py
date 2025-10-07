"""Registry of demo task app templates for `uvx synth-ai demo init`."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CopySpec:
    """File copy specification from repo-relative source to template-relative destination."""

    source: str
    destination: str
    make_executable: bool = False

    def absolute_source(self) -> Path:
        return (REPO_ROOT / self.source).resolve()


@dataclass(frozen=True)
class DemoTemplate:
    """Describes a demo task app template that can be materialised into the CWD."""

    template_id: str
    name: str
    description: str
    copy_specs: tuple[CopySpec, ...]
    default_subdir: str | None = None
    env_lines: tuple[str, ...] = ()
    config_source: str | None = None
    config_destination: str = "demo_config.toml"
    requires_modal: bool = False
    post_copy: Callable[[Path], None] | None = None

    def iter_copy_specs(self) -> Iterable[CopySpec]:
        return self.copy_specs

    def config_source_path(self) -> Path | None:
        if not self.config_source:
            return None
        return (REPO_ROOT / self.config_source).resolve()


DEMO_TEMPLATES: tuple[DemoTemplate, ...] = (
    DemoTemplate(
        template_id="math-modal",
        name="Math Single-Step (Modal deployment)",
        description="Packaged modal task app matching examples/rl math environment.",
        copy_specs=(
            CopySpec(
                "synth_ai/demos/demo_task_apps/math/modal_task_app.py",
                "task_app.py",
            ),
            CopySpec(
                "synth_ai/demos/demo_task_apps/math/deploy_task_app.sh",
                "deploy_task_app.sh",
                make_executable=True,
            ),
            CopySpec(
                "synth_ai/demos/demo_task_apps/math/config.toml",
                "configs/rl_from_base_qwen17.toml",
            ),
        ),
        default_subdir="math_demo",
        env_lines=(
            "# Required for task app auth to environment service",
            "ENVIRONMENT_API_KEY=",
            "",
            "# Optional: for CLI job submission and proxying OpenAI models",
            "SYNTH_API_KEY=",
            "OPENAI_API_KEY=",
            "",
            "# Optional: set to 'prod' to use production names",
            "ENVIRONMENT=",
        ),
        config_source="synth_ai/demos/demo_task_apps/math/config.toml",
        requires_modal=True,
        post_copy=lambda root: _postprocess_math_modal(root),
    ),
    DemoTemplate(
        template_id="crafter-local",
        name="Crafter GRPO (local FastAPI)",
        description="Lightweight wrapper around examples/warming_up_to_rl/task_app/grpo_crafter for local experimentation.",
        copy_specs=(
            CopySpec(
                "synth_ai/demos/demo_task_apps/crafter/grpo_crafter_task_app.py",
                "task_app.py",
            ),
            CopySpec(
                "synth_ai/demos/demo_task_apps/crafter/README.md",
                "README.md",
            ),
            CopySpec(
                "synth_ai/demos/demo_task_apps/crafter/configs/rl_from_base_qwen4b.toml",
                "configs/rl_from_base_qwen4b.toml",
            ),
            CopySpec(
                "synth_ai/demos/demo_task_apps/crafter/configs/crafter_fft_4b.toml",
                "configs/crafter_fft_4b.toml",
            ),
        ),
        default_subdir="crafter_demo",
        env_lines=(
            "ENVIRONMENT_API_KEY=",
            "SYNTH_API_KEY=",
            "",
            "# Optional: URL for existing Crafter task app",
            "TASK_APP_BASE_URL=",
        ),
        config_source="synth_ai/demos/demo_task_apps/crafter/configs/rl_from_base_qwen4b.toml",
        config_destination="demo_config.toml",
        requires_modal=False,
        post_copy=lambda root: _postprocess_crafter_local(root),
    ),
)


def get_demo_template(template_id: str) -> DemoTemplate | None:
    for template in DEMO_TEMPLATES:
        if template.template_id == template_id:
            return template
    return None


def list_demo_templates() -> tuple[DemoTemplate, ...]:
    return DEMO_TEMPLATES


def _postprocess_math_modal(root: Path) -> None:
    task_path = (root / "task_app.py").resolve()
    if not task_path.exists():
        return
    text = task_path.read_text(encoding="utf-8")
    text = text.replace('App("hendrycks-math-task-app")', 'App("hendrycks-math-task-app-demo")')
    text = text.replace('DEFAULT_TASK_APP_SECRET_NAME = "hendrycks-math-task-app-secret"', 'DEFAULT_TASK_APP_SECRET_NAME = "hendrycks-math-task-app-demo-secret"')
    task_path.write_text(text, encoding="utf-8")


CRAFT_DEMO_TEMPLATE = textwrap.dedent(
    '''
"""Demo-friendly wrapper for the GRPO Crafter task app."""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_ai.task.apps import ModalDeploymentConfig, registry
import sys
from pathlib import Path
_EXAMPLES_TASK_APP = Path(__file__).resolve().parents[4] / "examples" / "warming_up_to_rl" / "task_app"
if str(_EXAMPLES_TASK_APP) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_TASK_APP))
from grpo_crafter import build_config
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
    '''
).strip() + "\n"


def _postprocess_crafter_local(root: Path) -> None:
    task_path = (root / "task_app.py").resolve()
    if not task_path.exists():
        return
    task_path.write_text(CRAFT_DEMO_TEMPLATE, encoding="utf-8")

