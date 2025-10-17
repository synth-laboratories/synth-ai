import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _maybe_env() -> None:
    if os.getenv("SYNTH_API_KEY") and (
        os.getenv("PROD_BACKEND_URL") or os.getenv("BACKEND_BASE_URL") or os.getenv("SYNTH_BASE_URL")
    ):
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


def _deploy_or_lookup_task_app(tmp_dir: Path) -> str | None:
    """If TASK_APP_URL is unset, deploy or look up Modal URL and return it."""
    if os.getenv("TASK_APP_URL"):
        return os.getenv("TASK_APP_URL")

    envfile = tmp_dir / "modal.env"
    contents = []
    ak = os.getenv("SYNTH_API_KEY")
    if ak:
        contents.append(f"SYNTH_API_KEY={ak}")
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY") or ""
    if envk:
        contents.append(f"ENVIRONMENT_API_KEY={envk}")
    if contents:
        envfile.write_text("\n".join(contents) + "\n", encoding="utf-8")

    repo = _repo_root()
    cmd = [
        "uv",
        "run",
        "synth-ai",
        "task-app",
        "deploy",
        "grpo-crafter",
        "--name",
        "grpo-crafter-task-app",
    ]
    if envfile.exists():
        cmd.extend(["--env-file", str(envfile)])

    env = os.environ.copy()
    proc = subprocess.run(
        cmd, cwd=str(repo), text=True, capture_output=True, env=env, timeout=600
    )
    try:
        if envfile.exists():
            for line in envfile.read_text(encoding="utf-8").splitlines():
                if line.startswith("TASK_APP_BASE_URL="):
                    url = line.split("=", 1)[1].strip()
                    if url:
                        os.environ.setdefault("TASK_APP_URL", url)
                        return url
    except Exception:
        pass
    for line in (proc.stdout or "").splitlines():
        if "modal.run" in line:
            candidate = line.strip().split()[-1]
            if candidate.startswith("http"):
                os.environ.setdefault("TASK_APP_URL", candidate)
                return candidate
    return None


def test_examples_rl_run(tmp_path: Path) -> None:
    _maybe_env()
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    api_key = os.getenv("SYNTH_API_KEY")
    task_url = os.getenv("TASK_APP_URL") or _deploy_or_lookup_task_app(tmp_path)
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
