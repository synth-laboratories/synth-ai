#!/usr/bin/env python3
"""
Main entry point for the GRPO Synth Envs Hosted Service.

For local development:
    uvicorn main:app --reload --port 8000

For Modal deployment:
    modal deploy main.py
"""

from __future__ import annotations

import os

import modal

# Try to import Modal-specific features
try:
    from modal import App, Image, Volume, asgi_app

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

from synth_envs_hosted.hosted_app import create_app

# Local development mode
if __name__ == "__main__":
    import uvicorn

    # Create the FastAPI app
    app = create_app()

    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

# Modal deployment mode
elif MODAL_AVAILABLE:
    # Define Modal app
    modal_app = App("grpo-synth-envs-hosted")

    # Define the container image
    image = Image.debian_slim().pip_install(
        "fastapi",
        "uvicorn[standard]",
        "httpx",
        "pydantic",
        "synth-ai",
    )

    # Create or get the volume for state storage
    state_volume = Volume.from_name("synth-env-state", create_if_missing=True)

    # Define the ASGI app function
    @modal_app.function(
        image=image,
        min_containers=1,
        volumes={"/data/state": state_volume},
        secrets=[
            modal.Secret.from_name("vllm-config"),
        ],
    )
    @asgi_app()
    def fastapi_app():
        """Modal ASGI app factory."""
        return create_app()

    # Optional: Add a scheduled cleanup job
    @modal_app.function(
        schedule=modal.Period(hours=24),
        volumes={"/data/state": state_volume},
    )
    def cleanup_old_snapshots(max_age_hours: int = 48):
        """Periodic cleanup of old snapshots."""
        import shutil
        from datetime import datetime, timedelta
        from pathlib import Path

        base_path = Path("/data/state/runs")
        if not base_path.exists():
            return

        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        for run_dir in base_path.iterdir():
            if run_dir.is_dir():
                # Check modification time
                mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
                if mtime < cutoff_time:
                    print(f"Removing old run directory: {run_dir}")
                    shutil.rmtree(run_dir)

    # Export for Modal
    app = fastapi_app
