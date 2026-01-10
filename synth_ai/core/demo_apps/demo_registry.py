from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from synth_ai.core.paths import REPO_ROOT


@dataclass(slots=True)
class CopySpec:
    source: str
    destination: str
    make_executable: bool = False

    def absolute_source(self) -> Path:
        return (REPO_ROOT / self.source).resolve()


@dataclass(slots=True)
class DemoTemplate:
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
    default_secret_name: str | None = None

    def iter_copy_specs(self) -> Iterator[CopySpec]:
        yield from self.copy_specs

    def config_source_path(self) -> Path | None:
        if not self.config_source:
            return None
        return (REPO_ROOT / self.config_source).resolve()


def _postprocess_math_modal(root: Path) -> None:
    task_path = (root / "task_app.py").resolve()
    if not task_path.exists():
        return
    text = task_path.read_text(encoding="utf-8")
    text = text.replace('App("hendrycks-math-task-app")', 'App("hendrycks-math-task-app-demo")')
    text = text.replace(
        'DEFAULT_TASK_APP_SECRET_NAME = "hendrycks-math-task-app-secret"',
        'DEFAULT_TASK_APP_SECRET_NAME = "hendrycks-math-task-app-demo-secret"',
    )
    task_path.write_text(text, encoding="utf-8")


DEMO_TEMPLATES: dict[str, DemoTemplate] = {
    "math-modal": DemoTemplate(
        template_id="math-modal",
        name="Math Single-Step (Modal deployment)",
        description="Packaged Modal task app for the Hendrycks MATH single-step environment.",
        copy_specs=(
            CopySpec(
                "synth_ai/core/demo_apps/demo_task_apps/math/modal_task_app.py",
                "task_app.py",
            ),
            CopySpec(
                "synth_ai/core/demo_apps/demo_task_apps/math/config.toml",
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
        config_source="synth_ai/core/demo_apps/demo_task_apps/math/config.toml",
        requires_modal=True,
        post_copy=lambda root: _postprocess_math_modal(root),
        default_secret_name="hendrycks-math-task-app-demo-secret",
    ),
}


def list_demo_templates() -> tuple[DemoTemplate, ...]:
    return tuple(DEMO_TEMPLATES.values())


def get_demo_template(template_id: str) -> DemoTemplate | None:
    return DEMO_TEMPLATES.get(template_id)
