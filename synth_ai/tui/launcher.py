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
    api_key: str | None = None,
    refresh_interval: float = 5.0,
    event_interval: float = 2.0,
    limit: int = 50,
) -> None:
    """Launch the prompt learning monitoring TUI."""
    synth_key = api_key or get_api_key(required=False) or ""

    tui_root = Path(__file__).resolve().parent / "app"
    entry = tui_root / "dist" / "index.mjs"
    if not entry.exists():
        raise RuntimeError(
            "OpenTUI bundle not found. Build the TUI first:\n"
            "  cd synth_ai/tui/app\n"
            "  bun install\n"
            "  bun run build\n"
            f"Missing file: {entry}"
        )

    runtime = _find_runtime()
    if runtime is None:
        raise RuntimeError("Missing runtime. Install bun to run the TUI.")

    env = dict(os.environ)
    # URLs from urls.py (source of truth)
    env["SYNTH_BACKEND_URL"] = BACKEND_URL_BASE
    env["SYNTH_FRONTEND_URL"] = FRONTEND_URL_BASE
    # API key
    env["SYNTH_API_KEY"] = synth_key
    # TUI config
    if job_id:
        env["SYNTH_TUI_JOB_ID"] = job_id
    env["SYNTH_TUI_REFRESH_INTERVAL"] = str(refresh_interval)
    env["SYNTH_TUI_EVENT_INTERVAL"] = str(event_interval)
    env["SYNTH_TUI_LIMIT"] = str(limit)

    result = subprocess.run([runtime, str(entry)], env=env)
    # Exit silently regardless of how the TUI process ended
    # (could be user quit, backend disconnect, etc.)
    if result.returncode != 0:
        # Non-zero exit but don't raise - TUI lifecycle is complete
        pass


def _find_runtime() -> str | None:
    return which("bun")
