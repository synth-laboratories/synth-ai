import os
import re
import time
import json
import urllib.request
import urllib.error
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
    for candidate in (".env.test.prod", ".env.test.dev", ".env.test", ".env"):
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


def _deploy_or_lookup_task_app(tmp_dir: Path, *, backend: str | None = None) -> tuple[str | None, str, str]:
    """Deploy the Modal task app and return (url, stdout, stderr) for diagnostics."""
    envfile = tmp_dir / "modal.env"
    lines: list[str] = []
    if os.getenv("SYNTH_API_KEY"):
        lines.append(f"SYNTH_API_KEY={os.getenv('SYNTH_API_KEY')}")
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY") or ""
    if envk:
        lines.append(f"ENVIRONMENT_API_KEY={envk}")
    if backend:
        # Ensure deploy preflight uses the provided backend (belt-and-suspenders)
        lines.append(f"BACKEND_BASE_URL={backend}")
        lines.append(f"BACKEND_OVERRIDE={backend}")
        lines.append(f"PROD_BACKEND_URL={backend}")
        lines.append(f"SYNTH_BASE_URL={backend}")
    if lines:
        envfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    # Note: deploy subcommand does not accept --backend; backend is provided via env file

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True, env=env, timeout=600)

    # Preferred: CLI writes TASK_APP_BASE_URL into env file
    try:
        if envfile.exists():
            for line in envfile.read_text(encoding="utf-8").splitlines():
                if line.startswith("TASK_APP_BASE_URL="):
                    url = line.split("=", 1)[1].strip()
                    if url:
                        os.environ.setdefault("TASK_APP_URL", url)
                        return url, proc.stdout or "", proc.stderr or ""
    except Exception:
        pass

    # Fallback 1: parse stdout lines for a URL token
    for line in (proc.stdout or "").splitlines():
        if "modal.run" in line and "http" in line:
            parts = [p for p in line.strip().split() if p.startswith("http")]
            if parts:
                candidate = parts[-1]
                os.environ.setdefault("TASK_APP_URL", candidate)
                return candidate, proc.stdout or "", proc.stderr or ""

    # Fallback 2: handle wrapped URLs across lines by flattening whitespace
    try:
        import re as _re
        flat = "".join((proc.stdout or "").split())
        m = _re.search(r"https?://[A-Za-z0-9\-\.]+\.modal\.run", flat)
        if m:
            candidate = m.group(0)
            os.environ.setdefault("TASK_APP_URL", candidate)
            return candidate, proc.stdout or "", proc.stderr or ""
    except Exception:
        pass

    return None, proc.stdout or "", proc.stderr or ""


def _warmup_task_app(url: str, *, timeout_s: float = 300.0, interval_s: float = 2.0) -> None:
    """Poll the task app /health endpoint until it responds OK.

    Tries several header schemes using ENVIRONMENT_API_KEY/DEV_ENVIRONMENT_API_KEY.
    Accepts HTTP 200 or a JSON body that looks healthy. Ignores 'app is stopped' until warm.
    """
    key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY") or ""
    headers_variants = []
    if key:
        headers_variants.append({"X-API-Key": key})
        headers_variants.append({"Authorization": f"Bearer {key}"})
        headers_variants.append({"X-API-Keys": key})
    headers_variants.append({})  # try without auth as last resort

    base = url.rstrip("/")
    health_paths = ["/health", "/api/health", "/"]
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        for hdrs in headers_variants:
            for path in health_paths:
                health_url = base + path
                req = urllib.request.Request(health_url, method="GET")
                for k, v in hdrs.items():
                    req.add_header(k, v)
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        status = getattr(resp, "status", 200)
                        body = resp.read().decode("utf-8", errors="ignore")
                        if status == 200:
                            return
                        # Accept bodies that indicate health without strict 200
                        try:
                            data = json.loads(body)
                            if isinstance(data, dict) and (data.get("ok") or data.get("status") in ("ok", "healthy")):
                                return
                        except Exception:
                            pass
                        # If Modal returns app stopped, keep trying
                        if "app for invoked web endpoint is stopped" in body.lower():
                            last_err = body
                            continue
                except urllib.error.HTTPError as he:  # noqa: F841
                    # 404 during cold start is expected from Modal; retry
                    last_err = str(he)
                    continue
                except Exception as e:  # network or TLS, keep trying
                    last_err = str(e)
                    continue
        time.sleep(max(interval_s, 0.5))
    raise AssertionError(f"Task app failed to warm up at {base}; last_err={last_err}")


@pytest.mark.slow
def test_cli_train_rl_smoke(tmp_path: Path) -> None:
    _maybe_env()
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    # Provide a sensible default backend for CI if none provided
    if not backend:
        backend = "https://agent-learning.onrender.com/api"
    api_key = os.getenv("SYNTH_API_KEY")
    # Always deploy to ensure warm start, then warm up the endpoint
    task_url, deploy_out, deploy_err = _deploy_or_lookup_task_app(tmp_path, backend=backend)
    if not api_key:
        pytest.fail("SYNTH_API_KEY is required to run RL smoke test. Set it in env or .env.test.*")

    # Warm the web endpoint before CLI verification (avoid 'app is stopped')
    if not task_url:
        pytest.fail(
            "Task app deployment did not return a URL.\n"
            f"DEPLOY STDOUT:\n{deploy_out}\n\nDEPLOY STDERR:\n{deploy_err}\n"
        )
    _warmup_task_app(task_url, timeout_s=float(os.getenv("TASK_APP_WARMUP_TIMEOUT", "180")))

    poll_timeout = os.getenv("SYNTH_TRAIN_TEST_POLL_TIMEOUT", "180")
    poll_interval = os.getenv("SYNTH_TRAIN_TEST_POLL_INTERVAL", "10")

    envfile = tmp_path / "rl.env"
    env_lines = [f"SYNTH_API_KEY={api_key}", f"TASK_APP_URL={task_url}"]
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY")
    if envk:
        env_lines.append(f"ENVIRONMENT_API_KEY={envk}")
    envfile.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    repo = _repo_root()
    cfg = repo / "tests" / "artifacts" / "configs" / "rl.fft.small.toml"

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
            "CLI RL smoke failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    assert "✓ Job created" in proc.stdout
    assert _JOB_ID_PATTERN.search(proc.stdout), "job id not found in output"


