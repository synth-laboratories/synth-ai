"""Prompt learning TUI entrypoint (OpenTUI JS app)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import which

from synth_ai.core.env import get_api_key, get_backend_url
from synth_ai.sdk.api.train.utils import ensure_api_base


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
    backend = backend_base or get_backend_url()
    base_url = ensure_api_base(backend)
    synth_key = api_key or get_api_key(required=False) or ""

    tui_root = Path(__file__).resolve().parents[1] / "_tui"
    entry = tui_root / "dist" / "index.mjs"
    if not entry.exists():
        raise RuntimeError(
            "OpenTUI bundle not found. Build the TUI first:\n"
            "  cd synth_ai/_tui\n"
            "  bun install\n"
            "  bun run build\n"
            f"Missing file: {entry}"
        )

    runtime = _find_runtime()
    if runtime is None:
        raise RuntimeError("Missing runtime. Install bun to run the TUI.")

    env = dict(os.environ)
    env["SYNTH_TUI_API_BASE"] = base_url
    env["SYNTH_API_KEY"] = synth_key
    if job_id:
        env["SYNTH_TUI_JOB_ID"] = job_id
    env["SYNTH_TUI_REFRESH_INTERVAL"] = str(refresh_interval)
    env["SYNTH_TUI_EVENT_INTERVAL"] = str(event_interval)
    env["SYNTH_TUI_LIMIT"] = str(limit)

    subprocess.run([runtime, str(entry)], env=env, check=True)


def _find_runtime() -> str | None:
    return which("bun")
