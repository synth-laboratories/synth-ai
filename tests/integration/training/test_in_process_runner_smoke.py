"""Optional smoke tests for the in-process runner (tunneled task apps).

These are skipped unless RUN_INPROCESS_SMOKE=1 is set and backend/task-app keys
are available. Budgets are minimized to avoid long runs.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from synth_ai.sdk.task import run_in_process_job

pytestmark = [pytest.mark.integration, pytest.mark.requires_backend]


@pytest.mark.timeout(600)
def test_in_process_prompt_learning_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("RUN_INPROCESS_SMOKE") != "1":
        pytest.skip("Set RUN_INPROCESS_SMOKE=1 to exercise the smoke runner")

    api_key = os.getenv("SYNTH_API_KEY")
    env_key = os.getenv("ENVIRONMENT_API_KEY")
    if not api_key or not env_key:
        pytest.skip("SYNTH_API_KEY/ENVIRONMENT_API_KEY required for smoke run")

    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "synth_ai" / "cli" / "demo_apps" / "mipro" / "train_cfg.toml"
    task_app_path = repo_root / "synth_ai" / "cli" / "demo_apps" / "mipro" / "task_app.py"

    if not config_path.exists() or not task_app_path.exists():
        pytest.skip("Demo configs missing; skip smoke run")

    overrides = {
        # Minimize work so the smoke run finishes quickly
        "prompt_learning.mipro.num_iterations": 1,
        "prompt_learning.mipro.num_evaluations_per_iteration": 1,
        "prompt_learning.mipro.batch_size": 2,
        "prompt_learning.mipro.online_pool": [0, 1],
        "prompt_learning.mipro.bootstrap_train_seeds": [0, 1],
        "prompt_learning.policy.temperature": 0.0,
    }

    result = asyncio.run(
        run_in_process_job(
            job_type="prompt_learning",
            config_path=config_path,
            task_app_path=task_app_path,
            overrides=overrides,
            poll=False,  # submit-only for fast smoke
            timeout=300.0,
            health_check_timeout=60.0,
        )
    )

    assert result.job_id
    assert result.task_app_url
