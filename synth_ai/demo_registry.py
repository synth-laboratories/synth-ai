"""Registry of demo task app templates for `uvx synth-ai demo init`."""

from __future__ import annotations

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
        description="Lightweight wrapper around synth_ai.task.apps.grpo_crafter for local experimentation.",
        copy_specs=(
            CopySpec(
                "examples/warming_up_to_rl/task_app/grpo_crafter_task_app.py",
                "task_app.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/task_app/README.md",
                "README.md",
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
        config_source="examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml",
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


def _postprocess_crafter_local(root: Path) -> None:
    task_path = (root / "task_app.py").resolve()
    if not task_path.exists():
        return
    text = task_path.read_text(encoding="utf-8")
    sentinel = "return create_task_app(build_config())"
    replacement = "    config = build_config()\n    try:\n        config.app_id = f\"{config.app_id}-demo\"\n    except AttributeError:\n        pass\n    return create_task_app(config)"
    if sentinel in text:
        text = text.replace(sentinel, replacement)
    task_path.write_text(text, encoding="utf-8")
