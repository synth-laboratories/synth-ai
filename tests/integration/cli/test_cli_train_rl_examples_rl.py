import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _maybe_env() -> None:
    if os.getenv("SYNTH_API_KEY") and os.getenv("DEV_BACKEND_URL"):
        return
    for candidate in (".env.test.dev", ".env.test", ".env"):
        p = _repo_root() / candidate
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def test_examples_rl_run(tmp_path: Path) -> None:
    _maybe_env()
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("DEV_BACKEND_URL")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    api_key = os.getenv("SYNTH_API_KEY")
    task_url = os.getenv("TASK_APP_URL")
    if not backend or not api_key or not task_url:
        pytest.skip("SYNTH_API_KEY, DEV_BACKEND_URL, and TASK_APP_URL required for RL examples test")

    cfg = _repo_root() / "examples" / "rl" / "configs" / "rl_from_base_qwen.toml"

    cmd = [
        "uv",
        "run",
        "python",
        str(_repo_root() / "examples" / "rl" / "run_rl_and_save.py"),
        "--backend",
        backend,
        "--config",
        str(cfg),
        "--task-url",
        task_url,
    ]

    env = os.environ.copy()

    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        text=True,
        capture_output=True,
        env=env,
        timeout=360,
    )

    if proc.returncode != 0:
        pytest.fail(
            "examples/rl run failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
