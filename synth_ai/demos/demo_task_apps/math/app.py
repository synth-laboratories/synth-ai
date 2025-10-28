from __future__ import annotations

import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# Reuse the examples/rl task_app routes if available
try:
    from synth_ai.examples.rl.task_app import make_app as make_rl_app  # type: ignore
except Exception:  # fallback path when imported from repo root
    try:
        from examples.rl.task_app import make_app as make_rl_app  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Unable to import RL task app: {e}") from e


def create_app() -> FastAPI:
    # Configure math defaults via env (consumed by RL task_app helpers)
    os.environ.setdefault("DEMO_ENV_NAME", "math")
    os.environ.setdefault("DEMO_POLICY_NAME", "math-react")
    # Build base app
    app = make_rl_app()
    # CORS for local demo
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def run(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn

    uvicorn.run(create_app(), host=host, port=int(os.getenv("PORT", port)))
