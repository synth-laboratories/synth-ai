"""Lightweight Modal deploy wrapper for Banking77 task app (web)."""
from __future__ import annotations

import os
from pathlib import Path

try:
    import modal  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Modal is required to deploy: {exc}")

_here = Path(__file__).resolve()
_parents = list(_here.parents)
REPO_ROOT = _parents[3] if len(_parents) > 3 else Path.cwd()

app = modal.App("synth-banking77-web")

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "synth-ai",
        "datasets>=2.14.0",
        "fastapi>=0.115.0",
        "pydantic>=2.0.0",
        "httpx>=0.26.0",
        "python-dotenv>=1.0.0",
    )
    .env({"PYTHONPATH": "/opt/synth_ai_repo"})
    .add_local_dir(str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai", copy=True)
    .add_local_dir(str(REPO_ROOT / "examples"), "/opt/synth_ai_repo/examples", copy=True)
)
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    _image = _image.add_local_file(str(_env_file), "/opt/synth_ai_repo/.env")


@app.function(image=_image, timeout=600)
@modal.asgi_app()
def web():
    # Lazy import the task app to avoid local heavy deps
    import contextlib
    with contextlib.suppress(Exception):
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(str(REPO_ROOT / ".env"), override=False)
    from examples.task_apps.banking77.banking77_task_app import fastapi_app  # type: ignore
    return fastapi_app()
