"""Modal deployment helper for the Pokémon Showdown task app example.

This file mirrors the manual setup steps documented in the README:

- Clone `pokechamp` and install its Python dependencies.
- Clone the reference Pokémon Showdown server and install Node dependencies.
- Mount the local `synth-ai` repository so the task app code is available.

Deploy with:

```
modal deploy examples/task_apps/pokemon_battle/modal_app.py
```

After deployment the FastAPI service will be reachable at a URL similar to
`https://<org>--pokemon-showdown-task-app-example.modal.run`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
POKECHAMP_REPO = "https://github.com/sethkarten/pokechamp.git"
SHOWDOWN_REPO = "https://github.com/jakegrigsby/pokemon-showdown.git"

app = modal.App("pokemon-showdown-task-app-example")

BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "nodejs", "npm")
    .pip_install(["uvicorn[standard]", "fastapi", "httpx", "horizons-ai"])
    .run_commands(
        [
            "mkdir -p /external",
            f"git clone --depth 1 {POKECHAMP_REPO} /external/pokechamp || true",
            "pip install --no-cache-dir -r /external/pokechamp/requirements.txt",
            f"git clone --depth 1 {SHOWDOWN_REPO} /external/pokemon-showdown || true",
            "cd /external/pokemon-showdown && npm ci --no-optional",
        ]
    )
)

REPO_MOUNT = modal.Mount.from_local_dir(REPO_ROOT, remote_path="/workspace/synth-ai")


@app.function(
    image=BASE_IMAGE,
    mounts=[REPO_MOUNT],
    timeout=900,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("environment-api-key")],
    keep_warm=1,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the Synth task app via Modal."""

    import os
    from fastapi import APIRouter

    repo_path = Path("/workspace/synth-ai").resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    marker = Path("/tmp/.synth_ai_editable")
    if not marker.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_path)])
        marker.touch()

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("POKECHAMP_ROOT", "/external/pokechamp")
    os.environ.setdefault("POKEMON_SHOWDOWN_ROOT", "/external/pokemon-showdown")

    from examples.task_apps.pokemon_battle.task_app.pokemon_showdown import build_config
    from synth_ai.task.server import create_task_app

    app = create_task_app(build_config())

    health_router = APIRouter()

    @health_router.get("/healthz")
    def healthz():
        return {"status": "ok"}

    app.include_router(health_router)
    return app


@app.local_entrypoint()
def main():
    """Print handy commands for local testing."""

    print("Pokémon Showdown task app Modal helper")
    print("Deploy with: modal deploy examples/task_apps/pokemon_battle/modal_app.py")
    print("Test locally: modal serve examples/task_apps/pokemon_battle/modal_app.py")
    print("Once deployed, set TASK_APP_URL to the issued modal.run domain.")
