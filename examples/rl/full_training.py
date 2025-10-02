from __future__ import annotations

"""
examples/rl/full_training.py

End-to-end clustered RL training example using RlClient with robust polling.
"""

import argparse
import asyncio
import os
import json as _json
import httpx
from typing import Any, Dict, Optional

from synth_ai.learning import RlClient, JobHandle, stream_job_events
from synth_ai.config.base_url import get_backend_from_env

try:  # allow running without package context
    from examples.common.backend import resolve_backend_url as _resolve_backend_default  # type: ignore
except Exception:  # pragma: no cover - fallback for direct execution

    def _resolve_backend_default() -> str:
        base, _ = get_backend_from_env()
        base = base.rstrip("/")
        return base if base.endswith("/api") else f"{base}/api"


def _load_rl_env() -> None:
    """Load env from project .env, examples/rl/.env, and monorepo backend/.env.dev (best-effort)."""
    try:
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, "..", ".."))
        # Project and examples .env
        for p in (os.path.join(root, ".env"), os.path.join(here, ".env")):
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
        # Monorepo backend .env.dev
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


def _load_env(mode: str | None = None) -> tuple[str, str]:
    """Resolve backend base_url and api_key similar to openai_in_task_app.py.

    Precedence:
    - SYNTH_BACKEND_URL_OVERRIDE=local|dev|prod (preferred)
    - explicit mode arg (local|dev|prod)
    - default prod
    """
    # Load .env already done via _load_rl_env()

    override = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
    if override in {"local", "dev", "prod"}:
        try:
            import sys as _sys
            _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from common.backend import resolve_backend_url as _rb  # type: ignore
            base = _rb()
        except Exception:
            base = _resolve_backend_default()
        api_key = os.getenv("SYNTH_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing SYNTH_API_KEY in environment/.env")
        return base.rstrip("/"), api_key

    if mode is None:
        mode = os.getenv("SYNTH_MODE", "prod").strip().lower()
    if mode == "local":
        base_url = os.getenv("LOCAL_BACKEND_URL", "http://localhost:8000").strip()
        api_key = os.getenv("TESTING_LOCAL_SYNTH_API_KEY", os.getenv("SYNTH_API_KEY", "").strip()).strip()
        if not base_url:
            raise RuntimeError("Missing LOCAL_BACKEND_URL in environment/.env")
    elif mode == "dev":
        base_url = os.getenv("DEV_BACKEND_URL", "").strip()
        # Prefer SYNTH_API_KEY from synth-ai/.env if present
        api_key = os.getenv("SYNTH_API_KEY", "").strip() or os.getenv("DEV_SYNTH_API_KEY", "").strip()
        if not base_url or not api_key:
            raise RuntimeError("Missing DEV_BACKEND_URL or SYNTH_API_KEY in environment/.env")
    else:  # prod
        try:
            import sys as _sys
            _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from common.backend import resolve_backend_url as _rb  # type: ignore
            base_url = _rb()
        except Exception:
            base_url = _resolve_backend_default()
        api_key = os.getenv("SYNTH_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing SYNTH_API_KEY in environment/.env")
    return base_url.rstrip("/"), api_key


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
    # Deprecated: trainer start URL now resolved server-side
    p.add_argument("--trainer-start-url", type=str, default="")
    p.add_argument("--stream-seconds", type=int, default=None)
    p.add_argument("--empty-polls", type=int, default=None)
    p.add_argument("--startup-deadline-s", type=int, default=None)
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    # Fallback to env override if flags missing
    # Resolve like openai_in_task_app.py: prefer SYNTH_API_KEY from synth-ai/.env
    mode = os.getenv("SYNTH_MODE", None)
    try:
        backend_url, api_key = _load_env(mode)
    except Exception:
        # Fallback to prior behavior
        backend_url = args.backend_url or os.getenv("DEV_BACKEND_URL", "").strip() or os.getenv("PROD_BACKEND_URL", "").strip()
        if not backend_url:
            backend_url = _resolve_backend_default()
        backend_url = backend_url.rstrip("/")
        if not backend_url.endswith("/api"):
            backend_url = f"{backend_url}/api"
        api_key = args.api_key or os.getenv("SYNTH_API_KEY", "").strip() or os.getenv("BACKEND_API_KEY", "").strip()
    # Resolve Task App URL and Trainer ID like other scripts (may be overridden by TOML)
    task_app_url = (args.task_app_url or os.getenv("TASK_APP_BASE_URL", "").strip())
    if not task_app_url:
        task_app_url = os.getenv("LOCAL_TASK_APP_URL", "http://localhost:8001").strip()
    trainer_id = (args.trainer_id or os.getenv("TRAINER_ID", "").strip())

    # If trainer_id is missing, we'll use the direct OpenAI workflow path instead
    if not backend_url or not api_key or not task_app_url:
        print("Missing required flags: --backend-url, --api-key, --task-app-url")
        print(f"backend_url={'set' if backend_url else 'missing'} api_key={'set' if api_key else 'missing'}")
        print(f"task_app_url={'set' if task_app_url else 'missing'}")
        return 2
    inline_config: Optional[Dict[str, Any]] = None
    if args.config_path:
        try:
            import tomllib

            with open(args.config_path, "rb") as fh:
                inline_config = tomllib.load(fh)
        except Exception:
            pass
    # Merge config defaults with CLI (CLI overrides config)
    cfg_trainer = (inline_config or {}).get("trainer", {})
    cfg_job = (inline_config or {}).get("job", {})
    cfg_runner = (inline_config or {}).get("runner", {})
    model = args.model or cfg_job.get("model") or "Qwen/Qwen3-0.6B"
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_trainer.get("batch_size", 2))
    group_size = args.group_size if args.group_size is not None else int(cfg_trainer.get("group_size", 4))
    stream_seconds = args.stream_seconds if args.stream_seconds is not None else int(cfg_runner.get("stream_seconds", 0))
    # Use higher defaults to avoid premature assertion when backend events are delayed
    empty_polls = args.empty_polls if args.empty_polls is not None else int(cfg_runner.get("empty_polls_threshold", 120))
    startup_deadline_s = (
        args.startup_deadline_s if args.startup_deadline_s is not None else int(cfg_runner.get("startup_deadline_s", 180))
    )

    # Show effective configuration
    env_cfg = (inline_config or {}).get("env", {})
    svc_cfg = (inline_config or {}).get("services", {})
    # Prefer TOML services overrides if present
    try:
        if not args.task_app_url:
            v = (svc_cfg.get("task_url") or "").strip()
            if v:
                task_app_url = v
        # trainer_start_url ignored; server injects
    except Exception:
        pass
    env_max_steps = env_cfg.get("max_steps_per_episode")
    print(
        "Effective config:",
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
        },
    )

    # Branch: no trainer_id -> create RL job via /api/rl/jobs (server injects training_start_url)
    if not trainer_id:
        # Build body mirroring start_qwen_full_clustered.py semantics
        body: Dict[str, Any] = {
            "job_type": "rl",
            "data": {
                "model": model,
                "endpoint_base_url": task_app_url,
                **({"job_config_id": (args.job_config_id or None)} if args.job_config_id else {}),
                **({"config": inline_config} if inline_config else {}),
                "trainer": {"batch_size": batch_size, "group_size": group_size},
            },
        }
        # Ensure /api suffix
        api_base = backend_url.rstrip("/")
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
        except Exception as e:
            print(f"Create job error: {type(e).__name__}: {e}")
            return 1
        # Optional SSE (live events) like clustered starter
        sse_task = None
        if stream_seconds and int(stream_seconds) > 0:
            try:
                sse_task = asyncio.create_task(stream_job_events(backend_url, api_key, job_id, seconds=int(stream_seconds)))
            except Exception:
                sse_task = None
        # Poll with JobHandle to mirror robust polling used by clustered starter
        def _on_event(e: Dict[str, Any]) -> None:
            try:
                print(f"event seq={e.get('seq')} type={e.get('type')} msg={e.get('message')}")
            except Exception:
                pass
        def _on_metric(p: Dict[str, Any]) -> None:
            try:
                print(f"metric {p.get('name')} step={p.get('step')} epoch={p.get('epoch')} value={p.get('value')}")
            except Exception:
                pass
        handle = JobHandle(backend_url, api_key, job_id, strict=False, timeout=600.0)
        result = await handle.poll_until_terminal(
            interval_seconds=2.0,
            max_seconds=None,
            empty_polls_threshold=int(empty_polls),
            startup_deadline_s=int(startup_deadline_s),
            on_event=_on_event,
            on_metric=_on_metric,
        )
        print("Final:", result)
        # Ensure SSE task stops cleanly
        if sse_task:
            try:
                if not sse_task.done():
                    sse_task.cancel()
                try:
                    await asyncio.wait_for(sse_task, timeout=1.0)
                except Exception:
                    pass
            except Exception:
                pass
        return 0 if (result.get("status") == "succeeded") else 1

    # Trainer path (legacy/clustered):
    client = RlClient(backend_url, api_key)
    js = await client.create_job(
        model=model,
        task_app_url=task_app_url,
        trainer_id=trainer_id,
        trainer={"batch_size": batch_size, "group_size": max(2, group_size)},
        job_config_id=(args.job_config_id or None),
        inline_config=inline_config,
    )
    job_id = js.get("job_id") or js.get("id") or ""
    if not job_id:
        print("Failed to create job")
        return 1
    await client.start_job_if_supported(job_id)

    # Optional SSE in background
    sse_task = None
    if stream_seconds and int(stream_seconds) > 0:
        try:
            sse_task = asyncio.create_task(stream_job_events(backend_url, api_key, job_id, seconds=int(stream_seconds)))
        except Exception:
            sse_task = None

    def _on_event(e: Dict[str, Any]) -> None:
        try:
            print(f"event seq={e.get('seq')} type={e.get('type')} msg={e.get('message')}")
        except Exception:
            pass

    res = await client.poll_until_terminal(
        job_id,
        on_event=_on_event,
        on_metric=lambda p: print(f"metric {p.get('name')} step={p.get('step')} epoch={p.get('epoch')} value={p.get('value')}") if p else None,
        empty_polls_threshold=int(empty_polls),
        startup_deadline_s=int(startup_deadline_s),
    )
    print("Final:", res)
    if sse_task:
        try:
            await asyncio.wait_for(sse_task, timeout=1.0)
        except Exception:
            pass
    return 0 if (res.get("status") == "succeeded") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
