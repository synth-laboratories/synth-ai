"""TUI launcher - spawns the OpenTUI JS app via bun."""

import os
import subprocess
from pathlib import Path
from shutil import which

from synth_ai.core.env import get_api_key
from synth_ai.core.urls import BACKEND_URL_BASE, FRONTEND_URL_BASE


def run_prompt_learning_tui(
    *,
    job_id: str | None = None,
    backend_base: str | None = None,
    api_key: str | None = None,
    refresh_interval: float = 5.0,
    event_interval: float = 2.0,
    limit: int = 50,
) -> None:
    """Launch the prompt learning monitoring TUI."""
    synth_key = api_key or get_api_key(required=False) or ""

    tui_root = Path(__file__).resolve().parent / "app"
    entry = tui_root / "src" / "index.ts"
    if not entry.exists():
        raise RuntimeError(
            "OpenTUI entrypoint not found. Ensure the repo is intact:\n"
            "  cd synth_ai/tui/app\n"
            "  bun install\n"
            f"Missing file: {entry}"
        )

    runtime = _find_runtime()
    if runtime is None:
        raise RuntimeError("Missing runtime. Install bun to run the TUI.")

    # Ensure dependencies are installed
    _ensure_dependencies_installed(tui_root, runtime)

    env = dict(os.environ)
    # URLs from urls.py (source of truth), unless overridden
    env["SYNTH_BACKEND_URL"] = backend_base or BACKEND_URL_BASE
    env["SYNTH_FRONTEND_URL"] = FRONTEND_URL_BASE
    # API key
    env["SYNTH_API_KEY"] = synth_key
    # TUI config
    if job_id:
        env["SYNTH_TUI_JOB_ID"] = job_id
    env["SYNTH_TUI_REFRESH_INTERVAL"] = str(refresh_interval)
    env["SYNTH_TUI_EVENT_INTERVAL"] = str(event_interval)
    env["SYNTH_TUI_LIMIT"] = str(limit)

    result = subprocess.run([runtime, str(entry)], env=env, cwd=tui_root)
    # Exit silently regardless of how the TUI process ended
    # (could be user quit, backend disconnect, etc.)
    if result.returncode != 0:
        # Non-zero exit but don't raise - TUI lifecycle is complete
        pass


def _find_runtime() -> str | None:
    return which("bun")


def _ensure_dependencies_installed(tui_root: Path, runtime: str) -> None:
    """Ensure JavaScript dependencies are installed before running the TUI."""
    import shutil
    import sys

    package_json = tui_root / "package.json"
    package_dist_json = tui_root / "package.dist.json"
    node_modules = tui_root / "node_modules"

    # Check if package.json exists
    if not package_json.exists():
        raise RuntimeError(
            f"package.json not found in TUI app directory: {tui_root}\nEnsure the repo is intact."
        )

    # Check if node_modules exists (or if solid-js is installed as a quick check)
    if not node_modules.exists() or not (node_modules / "solid-js").exists():
        print("Installing TUI dependencies...", file=sys.stderr)
        sys.stderr.flush()

        # If package.dist.json exists, we're running from PyPI install and need to use it
        # The regular package.json has workspace refs that only work in dev
        use_dist = package_dist_json.exists()
        original_package_json = None

        if use_dist:
            # Backup and replace package.json with dist version
            original_package_json = package_json.read_text()
            shutil.copy(package_dist_json, package_json)

        try:
            # Run bun install
            install_result = subprocess.run(
                [runtime, "install"],
                cwd=tui_root,
                capture_output=True,
                text=True,
            )

            if install_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install TUI dependencies:\n"
                    f"stdout: {install_result.stdout}\n"
                    f"stderr: {install_result.stderr}\n"
                    f"Run manually: cd {tui_root} && bun install"
                )
        finally:
            # Restore original package.json if we swapped it
            if use_dist and original_package_json is not None:
                package_json.write_text(original_package_json)

        # Verify installation succeeded
        if not node_modules.exists() or not (node_modules / "solid-js").exists():
            raise RuntimeError(
                f"Dependencies installed but solid-js not found.\n"
                f"Run manually: cd {tui_root} && bun install"
            )
