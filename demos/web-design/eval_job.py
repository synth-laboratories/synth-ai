#!/usr/bin/env python3
"""Run ONE backend EvalJob for web-design (local backend) and print results.

This:
- starts the web-design task app on localhost:8103
- submits a single eval job to localhost:8000 with multiple seeds
- prints wall time, mean_score (reward), total_cost_usd, and per-seed breakdown
"""

import argparse
import os
import socket
import time
from pathlib import Path

import httpx
from synth_ai.core.env import mint_demo_api_key
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task.server import run_server_background


def _load_task_app_module() -> object:
    """Load the local task app module from this folder without sys.path hacks."""
    import importlib.util

    module_path = Path(__file__).resolve().with_name("web_design_task_app.py")
    spec = importlib.util.spec_from_file_location("web_design_task_app", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


BASELINE_STYLE_PROMPT = """You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy."""


def _parse_seeds(arg: str) -> list[int]:
    if "-" in arg:
        lo, hi = arg.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _pick_task_port(requested: int) -> int:
    """Pick a free port, starting from requested.

    If requested==0, start from 8103.
    """
    start = 8103 if int(requested) == 0 else int(requested)
    for p in range(start, start + 50):
        if _port_is_free(p):
            return p
    raise RuntimeError(f"No free port found in range [{start}, {start + 49}]")


def _task_app_healthy(url: str, env_api_key: str) -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{url}/health", headers={"X-API-Key": env_api_key})
            return r.status_code == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="http://localhost:8000",
        help="Backend base URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--task-port", type=int, default=8103, help="Task app port (default: 8103). Use 0 for auto."
    )
    parser.add_argument("--seeds", default="0-7", help="Seeds range/list (default: 0-7)")
    parser.add_argument(
        "--policy-model", default="gemini-2.5-flash-image", help="Image generation model"
    )
    parser.add_argument("--policy-provider", default="google", help="Provider for policy model")
    parser.add_argument(
        "--concurrency", type=int, default=2, help="Max concurrent rollouts (default: 2)"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Per-rollout timeout seconds (default: 600)"
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=2400.0,
        help="Overall eval job poll timeout (default: 2400)",
    )
    args = parser.parse_args()

    backend = str(args.backend).rstrip("/")

    synth_api_key = (os.environ.get("SYNTH_API_KEY") or "").strip()
    if not synth_api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        synth_api_key = mint_demo_api_key(backend_url=backend)
        os.environ["SYNTH_API_KEY"] = synth_api_key

    env_api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_api_key:
        env_api_key = ensure_localapi_auth(
            backend_base=backend,
            synth_api_key=synth_api_key,
        )
        os.environ["ENVIRONMENT_API_KEY"] = env_api_key

    requested_port = int(args.task_port)
    requested_port = 8103 if requested_port == 0 else requested_port
    requested_url = f"http://localhost:{requested_port}"

    # If port is busy but an existing task app is healthy, reuse it.
    if not _port_is_free(requested_port) and _task_app_healthy(requested_url, env_api_key):
        task_port = requested_port
        task_url = requested_url
        started_task_app = False
    else:
        task_port = _pick_task_port(int(args.task_port))
        task_url = f"http://localhost:{task_port}"
        started_task_app = True

    seeds = _parse_seeds(str(args.seeds))

    task_mod = _load_task_app_module()
    app_id = getattr(task_mod, "APP_ID", "web_design_generator")
    create_web_design_local_api = task_mod.create_web_design_local_api

    # Start task app (only if we aren't reusing an existing one)
    if started_task_app:
        app = create_web_design_local_api(BASELINE_STYLE_PROMPT)
        run_server_background(app, port=task_port)

    # Wait for task app readiness (auth required)
    auth_headers = {"X-API-Key": env_api_key}
    with httpx.Client(timeout=5.0) as client:
        for _ in range(40):
            try:
                r = client.get(f"{task_url}/health", headers=auth_headers)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Task app did not become healthy at {task_url}/health")

    if started_task_app:
        print(f"Task app started: {task_url} (port={task_port})")
    else:
        print(f"Task app already running: {task_url} (port={task_port})")

    # Build EvalJob (ONE job, many seeds)
    cfg = EvalJobConfig(
        task_app_url=task_url,
        backend_url=backend,
        api_key=synth_api_key,
        task_app_api_key=env_api_key,
        app_id=app_id,
        env_name="web_design",
        seeds=seeds,
        policy_config={
            "provider": str(args.policy_provider),
            "model": str(args.policy_model),
            "inference_mode": "synth_hosted",
        },
        verifier_config={
            "enabled": True,
            "reward_source": "verifier",
            "backend_base": backend,
            "backend_provider": "google",
            "backend_model": "gemini-2.5-flash",
            "verifier_graph_id": "zero_shot_verifier_rubric_single",
            "backend_outcome_enabled": True,
            "backend_event_enabled": False,
            "concurrency": 3,
            "timeout": 240.0,
            "weight_env": 0.0,
            "weight_event": 0.0,
            "weight_outcome": 1.0,
        },
        concurrency=int(args.concurrency),
        timeout=float(args.timeout),
    )

    t0 = time.time()
    # Submit with a larger timeout than the SDK default (local backends sometimes do preflight work).
    base = backend.rstrip("/")
    submit_url = f"{base}/api/eval/jobs" if not base.endswith("/api") else f"{base}/eval/jobs"
    job_request = {
        "task_app_url": cfg.task_app_url,
        "task_app_api_key": cfg.task_app_api_key,
        "app_id": cfg.app_id,
        "env_name": cfg.env_name,
        "seeds": cfg.seeds,
        "policy": dict(cfg.policy_config),
        "env_config": dict(cfg.env_config),
        "verifier_config": dict(cfg.verifier_config or {}),
        "max_concurrent": cfg.concurrency,
        "timeout": cfg.timeout,
    }

    with httpx.Client(timeout=httpx.Timeout(180.0)) as client:
        resp = client.post(
            submit_url,
            json=job_request,
            headers={
                "Authorization": f"Bearer {synth_api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        job_id = (resp.json() or {}).get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"No job_id in response: {resp.text[:500]}")

    print(f"EvalJob submitted: {job_id}")

    job = EvalJob.from_job_id(job_id, backend_url=backend, api_key=synth_api_key)
    result = job.poll_until_complete(timeout=float(args.poll_timeout), progress=True)
    wall_s = time.time() - t0

    print("\n" + "=" * 80)
    print("EVAL RESULTS (WEB DESIGN)")
    print("=" * 80)
    print(f"backend: {backend}")
    print(f"task_app_url: {task_url}")
    print(f"job_id: {job_id}")
    print(f"wall_time_s: {wall_s:.1f}")

    if result.succeeded:
        print(f"mean_reward: {result.mean_reward}")
        print(f"total_cost_usd: {result.total_cost_usd}")
        print(f"total_tokens: {result.total_tokens}")
        print(f"completed: {result.num_completed}/{result.num_total}")
        print("\nPer-seed:")
        for r in result.seed_results:
            seed = r.get("seed")
            reward = r.get("reward") or r.get("outcome_reward") or r.get("reward_mean") or r.get("score")
            cost = r.get("cost_usd")
            toks = r.get("tokens")
            lat = r.get("latency_ms")
            err = r.get("error")
            print(
                f"- seed={seed} reward={reward} cost_usd={cost} tokens={toks} latency_ms={lat} error={err}"
            )
        return 0

    print(f"status: {result.status.value}")
    print(f"error: {result.error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
