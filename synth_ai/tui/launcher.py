"""TUI launcher - spawns the OpenTUI JS app via bun."""

import os
import subprocess
from pathlib import Path
from shutil import which

from synth_ai.core.urls import BACKEND_URL_BASE, FRONTEND_URL_BASE

BUNTIME = which("bun")
TUI_ROOT_PATH = Path(__file__).resolve().parent / "app"
ENTRY_PATH = TUI_ROOT_PATH / "src" / "index.ts"


def _ensure_dependencies_installed() -> None:
    """Ensure JavaScript dependencies are installed before running the TUI."""
    import shutil
    import sys

    package_json = TUI_ROOT_PATH / "package.json"
    package_dist_json = TUI_ROOT_PATH / "package.dist.json"
    node_modules = TUI_ROOT_PATH / "node_modules"

    # Check if package.json exists
    if not package_json.exists():
        raise RuntimeError(
            f"package.json not found in TUI app directory: {TUI_ROOT_PATH}\nEnsure the repo is intact."
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
                [BUNTIME, "install"],
                cwd=TUI_ROOT_PATH,
                capture_output=True,
                text=True,
            )

            if install_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install TUI dependencies:\n"
                    f"stdout: {install_result.stdout}\n"
                    f"stderr: {install_result.stderr}\n"
                    f"Run manually: cd {TUI_ROOT_PATH} && bun install"
                )
        finally:
            # Restore original package.json if we swapped it
            if use_dist and original_package_json is not None:
                package_json.write_text(original_package_json)

        # Verify installation succeeded
        if not node_modules.exists() or not (node_modules / "solid-js").exists():
            raise RuntimeError(
                f"Dependencies installed but solid-js not found.\n"
                f"Run manually: cd {TUI_ROOT_PATH} && bun install"
            )


def run_tui() -> None:
    if not ENTRY_PATH.exists():
        raise RuntimeError(
            "OpenTUI entrypoint not found. Ensure the repo is intact:\n"
            "  cd synth_ai/tui/app\n"
            "  bun install\n"
            f"Missing file: {ENTRY_PATH}"
        )

    if BUNTIME is None:
        raise RuntimeError("Missing runtime. Install bun to run the TUI.")

    # Ensure dependencies are installed
    _ensure_dependencies_installed()

    env = dict(os.environ)
    env["SYNTH_BACKEND_URL"] = BACKEND_URL_BASE
    env["SYNTH_FRONTEND_URL"] = FRONTEND_URL_BASE
    if "SYNTH_API_KEY" not in env:
        env["SYNTH_API_KEY"] = ""
    env.setdefault("SYNTH_TUI_LAUNCH_CWD", os.getcwd())
    try:
        from synth_ai.sdk.opencode_skills import materialize_tui_opencode_config_dir

        opencode_config_dir = materialize_tui_opencode_config_dir(
            include_packaged_skills=["synth-api"]
        )
        env.setdefault("OPENCODE_CONFIG_DIR", str(opencode_config_dir))
    except Exception:
        opencode_config_dir = Path(__file__).resolve().parent / "opencode_config"
        if opencode_config_dir.exists():
            env.setdefault("OPENCODE_CONFIG_DIR", str(opencode_config_dir))

    cmd = [BUNTIME, str(ENTRY_PATH)]
    result = subprocess.run(cmd, env=env, cwd=TUI_ROOT_PATH)
    if result.returncode != 0:
        pass
