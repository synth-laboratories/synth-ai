"""
Integration tests for RL training with vision-language models.

Tests the full pipeline:
1. Deploy Crafter task app (same as SFT data collection)
2. Run RL training with Qwen3-VL-4B using image observations
3. Verify job creation and basic telemetry

The task app used here is the same one that generates SFT data with images,
ensuring consistency between offline (SFT) and online (RL) training.
"""

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
    """Load env vars from .env.test.* files if not already set."""
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
    """Deploy the Modal task app and return (url, stdout, stderr) for diagnostics.
    
    Uses the same grpo-crafter task app that generates vision SFT data.
    """
    envfile = tmp_dir / "modal.env"
    lines: list[str] = []
    if os.getenv("SYNTH_API_KEY"):
        lines.append(f"SYNTH_API_KEY={os.getenv('SYNTH_API_KEY')}")
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY") or ""
    if envk:
        lines.append(f"ENVIRONMENT_API_KEY={envk}")
    if backend:
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

    # Fallback 1: parse stdout for URL
    for line in (proc.stdout or "").splitlines():
        if "modal.run" in line and "http" in line:
            parts = [p for p in line.strip().split() if p.startswith("http")]
            if parts:
                candidate = parts[-1]
                os.environ.setdefault("TASK_APP_URL", candidate)
                return candidate, proc.stdout or "", proc.stderr or ""

    # Fallback 2: regex search
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
    
    Important for vision models as they may need longer warmup.
    """
    key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY") or ""
    headers_variants = []
    if key:
        headers_variants.append({"X-API-Key": key})
        headers_variants.append({"Authorization": f"Bearer {key}"})
        headers_variants.append({"X-API-Keys": key})
    headers_variants.append({})

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
                        # Accept bodies that indicate health
                        try:
                            data = json.loads(body)
                            if isinstance(data, dict) and (data.get("ok") or data.get("status") in ("ok", "healthy")):
                                return
                        except Exception:
                            pass
                        # Modal cold start
                        if "app for invoked web endpoint is stopped" in body.lower():
                            last_err = body
                            continue
                except urllib.error.HTTPError as he:
                    last_err = str(he)
                    continue
                except Exception as e:
                    last_err = str(e)
                    continue
        time.sleep(max(interval_s, 0.5))
    
    raise AssertionError(f"Task app failed to warm up at {base}; last_err={last_err}")


@pytest.mark.slow
@pytest.mark.vision
def test_cli_train_rl_vision_qwen3vl4b(tmp_path: Path) -> None:
    """Test RL training with Qwen3-VL-4B using vision-enabled task app.
    
    This test verifies the full vision RL pipeline:
    1. Task app deployment (same as used for SFT data collection)
    2. Policy uses image observations only (image_only_mode=true)
    3. RL training with vision-capable model
    4. Job creation and basic telemetry
    
    Marks:
        @pytest.mark.slow - Takes several minutes (task app deploy + warmup + job submit)
        @pytest.mark.vision - Requires vision model support
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    if not backend:
        backend = "https://agent-learning.onrender.com/api"
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.fail("SYNTH_API_KEY is required to run RL vision test. Set it in env or .env.test.*")

    # Deploy task app (same one used for SFT data collection)
    task_url, deploy_out, deploy_err = _deploy_or_lookup_task_app(tmp_path, backend=backend)
    if not task_url:
        pytest.fail(
            "Task app deployment did not return a URL.\n"
            f"DEPLOY STDOUT:\n{deploy_out}\n\nDEPLOY STDERR:\n{deploy_err}\n"
        )
    
    # Warm up task app (vision models may need longer)
    warmup_timeout = float(os.getenv("TASK_APP_WARMUP_TIMEOUT", "300"))  # 5min for vision
    _warmup_task_app(task_url, timeout_s=warmup_timeout)

    # Prepare env file for training
    poll_timeout = os.getenv("SYNTH_TRAIN_TEST_POLL_TIMEOUT", "180")
    poll_interval = os.getenv("SYNTH_TRAIN_TEST_POLL_INTERVAL", "10")

    envfile = tmp_path / "rl_vision.env"
    env_lines = [f"SYNTH_API_KEY={api_key}", f"TASK_APP_URL={task_url}"]
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY")
    if envk:
        env_lines.append(f"ENVIRONMENT_API_KEY={envk}")
    envfile.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # Use the vision RL config
    repo = _repo_root()
    cfg = repo / "examples" / "qwen_vl" / "configs" / "crafter_rl_vision_qwen3vl4b.toml"
    
    if not cfg.exists():
        pytest.fail(f"Vision RL config not found: {cfg}")

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
            "CLI RL vision test failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    # Verify job creation
    assert "✓ Job created" in proc.stdout or "job" in proc.stdout.lower(), \
        f"Job creation not confirmed in stdout:\n{proc.stdout}"
    
    # Verify job ID is present
    match = _JOB_ID_PATTERN.search(proc.stdout)
    assert match, f"job id not found in output:\n{proc.stdout}"
    
    job_id = match.group(1)
    print(f"✅ Vision RL job created: {job_id}")
    print(f"   Model: Qwen3-VL-4B")
    print(f"   Task App: {task_url}")
    print(f"   Image Mode: image_only")


@pytest.mark.slow
@pytest.mark.vision
def test_task_app_vision_support(tmp_path: Path) -> None:
    """Test that the Crafter task app supports vision observations.
    
    This test verifies that the same task app used for SFT data collection
    can provide image observations for RL training.
    
    Checks:
    - Task app deploys successfully
    - Health endpoint responds
    - Can make a rollout request with vision settings
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    if not backend:
        backend = "https://agent-learning.onrender.com/api"
    
    # Deploy task app
    task_url, deploy_out, deploy_err = _deploy_or_lookup_task_app(tmp_path, backend=backend)
    if not task_url:
        pytest.skip(
            "Task app deployment did not return a URL (may be expected in CI without Modal setup)\n"
            f"DEPLOY STDOUT:\n{deploy_out}\n\nDEPLOY STDERR:\n{deploy_err}\n"
        )
    
    # Warm up
    try:
        _warmup_task_app(task_url, timeout_s=120.0)
    except AssertionError as e:
        pytest.skip(f"Task app failed to warm up (may be expected in CI): {e}")
    
    # Test a simple rollout request with vision settings
    # This verifies the task app can handle vision policy config
    rollout_req = {
        "env_name": "crafter",
        "policy_name": "crafter-react",
        "seed": 0,
        "max_turns": 1,  # Just 1 step for quick test
        "env_config": {"max_steps_per_episode": 1},
        "policy_config": {
            "use_vision": True,
            "image_only_mode": True,
            "temperature": 0.6,
            "max_tokens": 128,
        },
        "inference_url": "test://dummy",  # Task app will handle this
        "return_trace": False,  # Don't need full trace for this test
    }
    
    rollout_url = task_url.rstrip("/") + "/rollout"
    req = urllib.request.Request(
        rollout_url,
        data=json.dumps(rollout_req).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            
            # Verify we got a response
            assert isinstance(result, dict), "Task app should return a dict"
            
            # May have error due to dummy inference URL, but that's OK
            # We just want to verify the task app accepts vision config
            print(f"✅ Task app supports vision config")
            print(f"   Response keys: {list(result.keys())}")
            
    except urllib.error.HTTPError as e:
        # Some errors are OK (e.g., inference failure with dummy URL)
        if e.code >= 500:
            pytest.fail(f"Task app returned server error: {e.code} - {e.read().decode('utf-8', errors='ignore')}")
        # 4xx errors might be expected with dummy inference URL
        print(f"⚠️  Got {e.code} error (may be expected with dummy inference URL)")
    except Exception as e:
        pytest.fail(f"Task app request failed: {e}")


@pytest.mark.slow
@pytest.mark.vision
def test_cli_train_rl_vision_small_config(tmp_path: Path) -> None:
    """Fast RL vision test using small config artifact.
    
    This test uses a minimal config (1 iteration, 3 steps, 1 episode)
    for fast CI verification of the vision RL pipeline.
    
    Faster than test_cli_train_rl_vision_qwen3vl4b but still validates:
    - Task app deployment
    - Vision policy configuration
    - Job submission and creation
    
    Marks:
        @pytest.mark.slow - Still takes a few minutes (deploy + warmup)
        @pytest.mark.vision - Requires vision model support
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    if not backend:
        backend = "https://agent-learning.onrender.com/api"
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.fail("SYNTH_API_KEY is required")

    # Deploy and warm up task app
    task_url, deploy_out, deploy_err = _deploy_or_lookup_task_app(tmp_path, backend=backend)
    if not task_url:
        pytest.fail(f"Task app deployment failed:\nSTDOUT:\n{deploy_out}\n\nSTDERR:\n{deploy_err}")
    
    warmup_timeout = float(os.getenv("TASK_APP_WARMUP_TIMEOUT", "180"))
    _warmup_task_app(task_url, timeout_s=warmup_timeout)

    # Prepare env
    poll_timeout = "120"  # Shorter for small config
    poll_interval = "5"
    
    envfile = tmp_path / "rl_vision_small.env"
    env_lines = [f"SYNTH_API_KEY={api_key}", f"TASK_APP_URL={task_url}"]
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY")
    if envk:
        env_lines.append(f"ENVIRONMENT_API_KEY={envk}")
    envfile.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # Use small artifact config for fast CI
    repo = _repo_root()
    cfg = repo / "tests" / "artifacts" / "configs" / "rl.vision.small.toml"
    
    if not cfg.exists():
        pytest.fail(f"Small vision RL config not found: {cfg}")

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
            "CLI RL vision small config test failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    # Verify job creation
    assert "✓ Job created" in proc.stdout or "job" in proc.stdout.lower()
    match = _JOB_ID_PATTERN.search(proc.stdout)
    assert match, f"job id not found in output:\n{proc.stdout}"
    
    job_id = match.group(1)
    print(f"✅ Fast vision RL job created: {job_id}")
    print(f"   Config: Small artifact (1 iter, 3 steps)")


if __name__ == "__main__":
    # For local testing
    import sys
    tmp = Path("/tmp/test_rl_vision")
    tmp.mkdir(exist_ok=True)
    
    print("Running vision RL tests locally...")
    print(f"Temp dir: {tmp}")
    
    try:
        test_task_app_vision_support(tmp)
        print("✅ test_task_app_vision_support passed")
    except Exception as e:
        print(f"❌ test_task_app_vision_support failed: {e}")
        sys.exit(1)
    
    try:
        test_cli_train_rl_vision_small_config(tmp)
        print("✅ test_cli_train_rl_vision_small_config passed")
    except Exception as e:
        print(f"❌ test_cli_train_rl_vision_small_config failed: {e}")
        sys.exit(1)
    
    print("\n🎉 All vision RL tests passed!")

