from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    deploy_script = root / "deploy_task_app.sh"
    if deploy_script.exists():
        import stat

        mode = deploy_script.stat().st_mode
        deploy_script.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


DEMO_TEMPLATES: dict[str, DemoTemplate] = {
    "math-modal": DemoTemplate(
        template_id="math-modal",
        name="Math Single-Step (Modal deployment)",
        description="Packaged modal task app matching examples/rl math environment.",
        copy_specs=(
            CopySpec(
                "synth_ai/demos/math/modal_task_app.py",
                "task_app.py",
            ),
            CopySpec(
                "synth_ai/demos/math/README.md",
                "README.md",
            ),
            CopySpec(
                "synth_ai/demos/math/deploy_task_app.sh",
                "deploy_task_app.sh",
                make_executable=True,
            ),
            CopySpec(
                "synth_ai/demos/math/config.toml",
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
        config_source="synth_ai/demos/math/config.toml",
        requires_modal=True,
        post_copy=lambda root: _postprocess_math_modal(root),
        default_secret_name="hendrycks-math-task-app-demo-secret",
    ),
    "crafter-local": DemoTemplate(
        template_id="crafter-local",
        name="Crafter GRPO (local FastAPI)",
        description="Lightweight wrapper around examples/warming_up_to_rl/task_app/grpo_crafter for local experimentation.",
        copy_specs=(
            CopySpec(
                "synth_ai/demos/crafter/grpo_crafter_task_app.py",
                "task_app.py",
            ),
            CopySpec(
                "synth_ai/demos/crafter/README.md",
                "README.md",
            ),
            CopySpec(
                "synth_ai/demos/crafter/configs/rl_from_base_qwen4b.toml",
                "configs/rl_from_base_qwen4b.toml",
            ),
            CopySpec(
                "synth_ai/demos/crafter/configs/crafter_fft_4b.toml",
                "configs/crafter_fft_4b.toml",
            ),
            CopySpec(
                "examples/warming_up_to_rl/task_app/grpo_crafter.py",
                "grpo_crafter.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/task_app/synth_envs_hosted",
                "synth_envs_hosted",
            ),
            CopySpec(
                "examples/warming_up_to_rl/run_local_rollout.py",
                "run_local_rollout.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/run_local_rollout_traced.py",
                "run_local_rollout_traced.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/shared.py",
                "shared.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/export_trace_sft.py",
                "export_trace_sft.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/run_fft_and_save.py",
                "run_fft_and_save.py",
            ),
            CopySpec(
                "examples/warming_up_to_rl/run_local_rollout_modal.py",
                "run_local_rollout_modal.py",
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
        default_secret_name="grpo-crafter-demo-secret",
    ),
}


def list_demo_templates() -> tuple[DemoTemplate, ...]:
    return tuple(DEMO_TEMPLATES.values())


def get_demo_template(template_id: str) -> DemoTemplate | None:
    return DEMO_TEMPLATES.get(template_id)
