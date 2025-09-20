from __future__ import annotations

"""
examples/rl/run_rl_job.py

Refactored to use synth_ai.learning abstractions:
- RlClient for job creation/start
- JobHandle for robust polling (since_seq, linked_job_id, metrics)
- stream_job_events for optional SSE
- backend_health and task_app_health for diagnostics
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

import httpx
from synth_ai.learning import (
    RlClient,
    JobHandle,
    stream_job_events,
    backend_health,
    task_app_health,
    validate_task_app_url,
)
from synth_ai.config.base_url import get_backend_from_env


def _load_rl_env() -> None:
    """Load env from project .env, examples/rl/.env, and monorepo backend/.env.dev (best-effort)."""
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # 1) Project and examples .env
        for p in (os.path.join(root, ".env"), os.path.join(os.path.dirname(__file__), ".env")):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        # 2) Monorepo backend .env.dev
        monorepo_env = os.path.join(root, "..", "monorepo", "backend", ".env.dev")
        if os.path.exists(monorepo_env):
            with open(monorepo_env, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    _load_rl_env()
    p = argparse.ArgumentParser()
    p.add_argument("--backend-url", type=str, default=os.getenv("PROD_BACKEND_URL", "").strip())
    p.add_argument("--api-key", type=str, default=os.getenv("SYNTH_API_KEY", "").strip())
    p.add_argument("--task-app-url", type=str, default=os.getenv("TASK_APP_BASE_URL", "").strip())
    p.add_argument("--trainer-id", type=str, default=os.getenv("TRAINER_ID", "").strip())
    p.add_argument("--model", type=str, default=os.getenv("QWEN_MODEL", "").strip() or None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--job-config-id", type=str, default=os.getenv("JOB_CONFIG_ID", "").strip())
    p.add_argument("--config-path", type=str, default=os.getenv("RL_CONFIG_PATH", "examples/rl/crafter_online.toml").strip())
    p.add_argument("--job-id", type=str, default=os.getenv("RL_JOB_ID", "").strip())
    p.add_argument("--stream-seconds", type=int, default=None)
    p.add_argument("--timeout", type=float, default=float(os.getenv("HTTP_TIMEOUT", "600") or 600.0))
    p.add_argument("--empty-polls", type=int, default=None)
    p.add_argument("--startup-deadline-s", type=int, default=None)
    # Deprecated: trainer start URL now resolved server-side
    p.add_argument("--trainer-start-url", type=str, default="")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    # Fallback to env override if flags missing
    base = (args.backend_url or "").rstrip("/")
    api_key = args.api_key
    if not base or not api_key:
        env_base, env_key = get_backend_from_env()
        base = base or env_base
        api_key = api_key or env_key
    if not base or not api_key:
        print("Missing --backend-url and/or --api-key (or environment)")
        return 2
    # Defer task app URL validation until after we apply config fallbacks

    # Health checks (best-effort)
    try:
        await backend_health(base, api_key)
    except Exception:
        pass
    if args.task_app_url:
        try:
            await task_app_health(args.task_app_url)
        except Exception:
            pass

    # Load inline config if provided
    inline_config: Optional[Dict[str, Any]] = None
    if args.config_path:
        try:
            import tomllib

            with open(args.config_path, "rb") as fh:
                inline_config = tomllib.load(fh)
            print(f"Loaded inline RL config from {args.config_path}")
        except Exception as e:
            print(f"Failed to load config TOML: {e}")
            return 2

    # Merge config defaults with CLI (CLI overrides config)
    cfg_trainer = (inline_config or {}).get("trainer", {})
    cfg_job = (inline_config or {}).get("job", {})
    cfg_runner = (inline_config or {}).get("runner", {})
    model = args.model or cfg_job.get("model") or "Qwen/Qwen3-0.6B"
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_trainer.get("batch_size", 2))
    group_size = args.group_size if args.group_size is not None else int(cfg_trainer.get("group_size", 4))
    stream_seconds = args.stream_seconds if args.stream_seconds is not None else int(cfg_runner.get("stream_seconds", 0))
    empty_polls = args.empty_polls if args.empty_polls is not None else int(cfg_runner.get("empty_polls_threshold", 5))
    startup_deadline_s = (
        args.startup_deadline_s if args.startup_deadline_s is not None else int(cfg_runner.get("startup_deadline_s", 45))
    )

    # Apply config fallbacks for service URLs (task app, trainer start) before validation
    env_cfg = (inline_config or {}).get("env", {})
    svc_cfg = (inline_config or {}).get("services", {})
    # Prefer TOML services overrides if CLI/env not set
    try:
        if not args.task_app_url:
            v = (svc_cfg.get("task_url") or "").strip()
            if v:
                os.environ.setdefault("TASK_APP_BASE_URL", v)
                args.task_app_url = v
        if not args.trainer_start_url:
            v2 = (svc_cfg.get("trainer_start_url") or "").strip()
            if v2:
                os.environ.setdefault("TRAINER_START_URL", v2)
                args.trainer_start_url = v2
    except Exception:
        pass
    # Validate presence of task app URL for new jobs (not required when attaching to an existing job)
    if not args.job_id:
        if not args.task_app_url:
            print("Missing --task-app-url for job creation")
            return 2
        validate_task_app_url(args.task_app_url, name="TASK_APP_BASE_URL")

    # Show effective configuration
    env_max_steps = env_cfg.get("max_steps_per_episode")
    print(
        "Effective config:",
        json.dumps(
            {
                "config_path": args.config_path or None,
                "model": model,
                "trainer": {"batch_size": batch_size, "group_size": group_size},
                "env": {"max_steps_per_episode": env_max_steps},
                "runner": {
                    "stream_seconds": stream_seconds,
                    "empty_polls_threshold": empty_polls,
                    "startup_deadline_s": startup_deadline_s,
                },
            }
        ),
    )

    job_id = args.job_id.strip()
    if not job_id:
        # Branch: create RL job via backend /rl/jobs if trainer_id omitted
        if not args.trainer_id:
            validate_task_app_url(args.task_app_url, name="TASK_APP_BASE_URL")
            # Load inline config
            inline_config: Optional[Dict[str, Any]] = None
            cfg_path = (args.config_path or "").strip()
            if cfg_path:
                try:
                    import tomllib as _toml
                    with open(cfg_path, "rb") as fh:
                        inline_config = _toml.load(fh)
                    print(f"Loaded inline RL config from {cfg_path}")
                except Exception as e:
                    print(f"Failed to load config TOML: {e}")
                    return 2
            # Build body; mirror start_qwen_full_clustered.py
            # Compute a sane default trainer start URL: POST to the task app base ("/")
            default_trainer_start = (args.task_app_url or "").rstrip("/") + "/"
            trainer_start_url = (args.trainer_start_url or default_trainer_start).rstrip("/") + "/"

            body: Dict[str, Any] = {
                "job_type": "rl",
                "data": {
                    "model": model,
                    "endpoint_base_url": (args.task_app_url or "").rstrip("/"),
                    **({"job_config_id": (args.job_config_id or None)} if args.job_config_id else {}),
                    **({"config": inline_config} if inline_config else {}),
                    "trainer": {"batch_size": batch_size, "group_size": group_size},
                    # Force correct trainer start endpoint (prod): POST "/" on the task app base
                    "training_start_url": trainer_start_url,
                },
            }
            # POST to /api/rl/jobs (ensure /api suffix)
            api_base = base.rstrip("/")
            if not api_base.endswith("/api"):
                api_base = f"{api_base}/api"
            url = f"{api_base}/rl/jobs"
            try:
                import uuid as _uuid
                headers = {"Idempotency-Key": _uuid.uuid4().hex, "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                timeouts = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=60.0)
                async with httpx.AsyncClient(timeout=timeouts) as client:
                    resp = await client.post(url, json=body, headers=headers)
                txt = (resp.text or "")[:2000]
                print(f"HTTP {resp.status_code}\n{txt}")
                if resp.status_code not in (200, 201):
                    return 1
                js = resp.json()
                job_id = js.get("job_id") or js.get("id") or ""
                if not job_id:
                    print("Warning: no job id in response")
                    return 1
                print(f"Created job: {job_id}")
            except Exception as e:
                print(f"Create job error: {type(e).__name__}: {e}")
                return 1
        else:
            # Legacy trainer path
            client = RlClient(base, api_key, timeout=args.timeout)
            js = await client.create_job(
                model=model,
                task_app_url=args.task_app_url,
                trainer_id=args.trainer_id,
                trainer={"batch_size": batch_size, "group_size": group_size},
                job_config_id=(args.job_config_id or None),
                inline_config=inline_config,
            )
            job_id = js.get("job_id") or js.get("id") or ""
            if not job_id:
                print("Failed to create job")
                return 1
            await client.start_job_if_supported(job_id)
    else:
        print(f"Attached to existing job: {job_id}")

    # Terminal event fast-exit support
    TERMINAL_EVENTS = {
        "workflow.completed",
        "workflow.failed",
        "workflow.cancelled",
        "workflow.termination.signal",
        "rl.job.completed",
        "rl.job.failed",
        "rl.finalization.done",
        "rl.finalization.signal",
        # Treat train-completed as terminal for client purposes
        "rl.train.completed",
        "rl.train.failed",
    }
    terminal_flag: asyncio.Event = asyncio.Event()
    # Global watchdog to prevent indefinite stalls (env override RL_CLIENT_MAX_POLL_SECONDS)
    try:
        watchdog_seconds = float(os.getenv("RL_CLIENT_MAX_POLL_SECONDS", "0") or 0)
    except Exception:
        watchdog_seconds = 0.0
    watchdog_task = None
    if watchdog_seconds and watchdog_seconds > 0:
        watchdog_task = asyncio.create_task(asyncio.sleep(watchdog_seconds))

    # Start background SSE to detect terminal events quickly (even if stream_seconds==0, force minimal streaming)
    async def _on_sse(e: Dict[str, Any]) -> None:
        try:
            et = str(e.get("type") or "")
            print(f"[sse] {e.get('seq')} {et}: {e.get('message')}")
            if et in TERMINAL_EVENTS:
                terminal_flag.set()
        except Exception:
            pass
    # Ensure SSE runs long enough to capture terminal events (overridable via env)
    try:
        _sse_floor = int(os.getenv("RL_CLIENT_SSE_SECONDS", "1800") or 1800)
    except Exception:
        _sse_floor = 1800
    sse_seconds = max(int(stream_seconds or 0), _sse_floor)
    sse_task = asyncio.create_task(stream_job_events(base, api_key, job_id, seconds=sse_seconds, on_event=_on_sse))

    # Give SSE a brief head start to catch immediate terminal events
    try:
        await asyncio.wait_for(terminal_flag.wait(), timeout=0.1)
        if sse_task:
            try:
                sse_task.cancel()
            except Exception:
                pass
        print("Final: {\"status\": \"succeeded\", \"reason\": \"terminal_event_pre\"}")
        return 0
    except asyncio.TimeoutError:
        pass

    # Poll until terminal
    def _on_event(e: Dict[str, Any]) -> None:
        try:
            et = str(e.get("type") or "")
            print(f"event seq={e.get('seq')} type={et} msg={e.get('message')}")
            if et in TERMINAL_EVENTS:
                terminal_flag.set()
        except Exception:
            pass
    def _on_metric(p: Dict[str, Any]) -> None:
        try:
            print(f"metric {p.get('name')} step={p.get('step')} epoch={p.get('epoch')} value={p.get('value')}")
        except Exception:
            pass
    # If created via RL jobs route, fall back to standard JobHandle polling (race with terminal SSE)
    if not args.trainer_id:
        handle = JobHandle(base, api_key, job_id, strict=False, timeout=args.timeout)
        # Allow poll interval override via env for quicker detection
        try:
            _poll_iv = float(os.getenv("RL_CLIENT_POLL_INTERVAL", "1.0") or 1.0)
        except Exception:
            _poll_iv = 1.0
        poll_task = asyncio.create_task(
            handle.poll_until_terminal(
                interval_seconds=_poll_iv,
                max_seconds=None,
                empty_polls_threshold=int(empty_polls),
                startup_deadline_s=int(startup_deadline_s),
                on_event=_on_event,
                on_metric=_on_metric,
            )
        )
        wait_set = {poll_task, asyncio.create_task(terminal_flag.wait())}
        if watchdog_task:
            wait_set.add(watchdog_task)
        done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        if terminal_flag.is_set() and not poll_task.done():
            # Fast-exit on terminal event
            try:
                poll_task.cancel()
            except Exception:
                pass
            if sse_task:
                try:
                    sse_task.cancel()
                except Exception:
                    pass
            print("Final: {\"status\": \"succeeded\", \"reason\": \"terminal_event\"}")
            return 0
        if watchdog_task and watchdog_task in done and not poll_task.done():
            try:
                poll_task.cancel()
            except Exception:
                pass
            if sse_task:
                try:
                    sse_task.cancel()
                except Exception:
                    pass
            print("Final: {\"status\": \"timeout\"}")
            return 1
        # Otherwise use poll result
        result = await poll_task
        print(f"Final: {json.dumps(result)}")
        return 0 if (result.get("status") == "succeeded") else 1

    # Trainer job: use standard JobHandle polling
    handle = JobHandle(base, api_key, job_id, strict=False, timeout=args.timeout)
    poll_task2 = asyncio.create_task(
        handle.poll_until_terminal(
            interval_seconds=2.0,
            max_seconds=None,
            empty_polls_threshold=int(empty_polls),
            startup_deadline_s=int(startup_deadline_s),
            on_event=_on_event,
            on_metric=_on_metric,
        )
    )
    wait_set2 = {poll_task2, asyncio.create_task(terminal_flag.wait())}
    if watchdog_task:
        wait_set2.add(watchdog_task)
    done2, pending2 = await asyncio.wait(wait_set2, return_when=asyncio.FIRST_COMPLETED)
    if terminal_flag.is_set() and not poll_task2.done():
        try:
            poll_task2.cancel()
        except Exception:
            pass
        if sse_task:
            try:
                sse_task.cancel()
            except Exception:
                pass
        print("Final: {\"status\": \"succeeded\", \"reason\": \"terminal_event\"}")
        result = {"status": "succeeded"}
    elif watchdog_task and watchdog_task in done2 and not poll_task2.done():
        try:
            poll_task2.cancel()
        except Exception:
            pass
        if sse_task:
            try:
                sse_task.cancel()
            except Exception:
                pass
        print("Final: {\"status\": \"timeout\"}")
        result = {"status": "timeout"}
    else:
        result = await poll_task2
    print(f"Final: {json.dumps(result)}")
    if sse_task:
        try:
            sse_task.cancel()
            await asyncio.wait_for(sse_task, timeout=0.5)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
