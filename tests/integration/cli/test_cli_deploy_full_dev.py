import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests


pytestmark = pytest.mark.integration


def _have_modal_cli() -> bool:
    return shutil.which("modal") is not None


def _have_modal_auth_env() -> bool:
    # Require token-based auth via environment to avoid interactive prompts
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))


@pytest.mark.skipif(not (_have_modal_cli() and _have_modal_auth_env()), reason="modal CLI not installed or not authenticated via env vars")
def test_deploy_and_fetch_task_info(tmp_path: Path):
    env = os.environ.copy()
    env.setdefault("ENVIRONMENT_API_KEY", "test_env_key_123")
    # Ensure backend preflight is skipped in CI/local without backend configured
    env.pop("SYNTH_API_KEY", None)
    env.pop("BACKEND_BASE_URL", None)
    env.pop("SYNTH_BASE_URL", None)
    repo_root = Path(__file__).resolve().parents[3]
    # Use a temp env file so the deploy command can save TASK_APP_BASE_URL into it
    tmp_env = tmp_path / ".env"
    # Provide minimal env so CLI can load values and avoid interactive prompts
    tmp_env.write_text(
        "\n".join(
            [
                f"ENVIRONMENT_API_KEY={env['ENVIRONMENT_API_KEY']}",
                "OPENAI_API_KEY=dummy",
                "GROQ_API_KEY=dummy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Real deploy (no dry-run); rely on Modal CLI being configured in the environment
    cp = subprocess.run(
        [
            "uv",
            "run",
            "synth-ai",
            "deploy",
            "math-single-step",
            "--env-file",
            str(tmp_env),
        ],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(repo_root),
        timeout=600,
    )
    # Show output to aid debugging on CI failures
    print(cp.stdout)
    print(cp.stderr)
    assert cp.returncode == 0

    # Prefer URL written into the env file; fallback to parsing stdout
    base_url = None
    try:
        for line in tmp_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TASK_APP_BASE_URL="):
                base_url = line.split("=", 1)[1].strip()
                break
    except Exception:
        pass
    if not base_url:
        m = re.search(r"https?://[^\s]+modal\.run[^\s]*", cp.stdout)
        base_url = m.group(0) if m else None
    assert base_url, "Could not determine deployed task app URL"

    # Wait for the deployed app to be reachable
    start = time.time()
    last_err: Exception | None = None
    while time.time() - start < 180:
        try:
            r = requests.get(base_url.rstrip("/") + "/health", timeout=5)
            # 200 or 400 means the service is up (400 when auth missing)
            if r.status_code in (200, 400):
                break
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    else:
        raise AssertionError(f"Deployed app never became healthy: {last_err}")

    # Fetch task_info with auth
    headers = {"X-API-Key": env["ENVIRONMENT_API_KEY"]}
    info = requests.get(base_url.rstrip("/") + "/task_info", headers=headers, timeout=15)
    assert 200 <= info.status_code < 300
    js = info.json()
    assert isinstance(js, dict) and any(
        k in js for k in ("task", "taskset", "dataset", "inference", "capabilities")
    )

