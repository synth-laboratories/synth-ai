import os
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_JOB_ID_PATTERN = re.compile(r"job[_-]id\s*[:=]\s*([a-zA-Z0-9_-]+)")


def _maybe_env() -> None:
    if os.getenv("SYNTH_API_KEY") and (
        os.getenv("DEV_BACKEND_URL") or os.getenv("BACKEND_BASE_URL") or os.getenv("SYNTH_BASE_URL")
    ):
        return
    repo = _repo_root()
    for candidate in (".env.test.dev", ".env.test", ".env"):
        p = repo / candidate
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            continue


def _deploy_or_lookup_task_app(tmp_dir: Path) -> str | None:
    """If TASK_APP_URL is unset, deploy or look up Modal URL and return it."""
    if os.getenv("TASK_APP_URL"):
        return os.getenv("TASK_APP_URL")

    # Prepare env file for the CLI to write TASK_APP_BASE_URL into
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
    # Best-effort parse: CLI writes TASK_APP_BASE_URL to the first env file
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
    # Fallback: parse stdout for a printed URL
    for line in (proc.stdout or "").splitlines():
        if "modal.run" in line:
            candidate = line.strip().split()[-1]
            if candidate.startswith("http"):
                os.environ.setdefault("TASK_APP_URL", candidate)
                return candidate
    return None


@pytest.mark.fast
def test_cli_train_rl_small_configs(tmp_path: Path) -> None:
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
        pytest.skip("SYNTH_API_KEY, backend base URL, and TASK_APP_URL required for RL CLI integration test")

    poll_timeout = os.getenv("SYNTH_TRAIN_TEST_POLL_TIMEOUT", "240")
    poll_interval = os.getenv("SYNTH_TRAIN_TEST_POLL_INTERVAL", "10")

    envfile = tmp_path / "rl.env"
    envfile.write_text(f"SYNTH_API_KEY={api_key}\nTASK_APP_URL={task_url}\n", encoding="utf-8")

    repo = _repo_root()
    configs = [
        (repo / "tests" / "artifacts" / "configs" / "rl.fft.small.toml"),
        (repo / "tests" / "artifacts" / "configs" / "rl.lora.small.toml"),
    ]

    for cfg in configs:
        cmd = [
            "uvx",
            "synth-ai",
            "train",
            "--type",
            "rl",
            "--config",
            str(cfg),
            "--backend",
            backend,
            "--task-url",
            task_url,
            "--env-file",
            str(envfile),
            "--no-poll",
            "--poll-timeout",
            poll_timeout,
            "--poll-interval",
            poll_interval,
        ]

        env = os.environ.copy()
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            text=True,
            capture_output=True,
            env=env,
            timeout=int(float(poll_timeout)) + 60,
        )
        if proc.returncode != 0:
            pytest.fail(
                "CLI RL train failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        assert "✓ Job created" in proc.stdout
        assert _JOB_ID_PATTERN.search(proc.stdout), "job id not found in output"


