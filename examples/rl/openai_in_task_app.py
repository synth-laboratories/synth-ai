from __future__ import annotations

"""
examples/rl/openai_in_task_app.py

Create and monitor a direct rollout where the Task App calls OpenAI directly via backend workflow TaskAppOpenAIWorkflow.
Requires TASK_APP_BASE_URL to point to the user’s Modal app with OPENAI_API_KEY set.
"""

import argparse
import asyncio
import os
from typing import Any, Dict, Optional

import json
import httpx


def _load_rl_env() -> None:
    """Load key=value pairs from project .env and examples/rl/.env into os.environ before arg parsing."""
    try:
        # 1) Project root .env (holds SYNTH_API_KEY)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for p in (os.path.join(root, ".env"), os.path.join(os.path.dirname(__file__), ".env")):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key:
                        os.environ[key] = val
        # 2) Also try to load from monorepo backend .env.dev for JWT_SECRET_KEY
        monorepo_env = os.path.join(root, "..", "monorepo", "backend", ".env.dev")
        if os.path.exists(monorepo_env):
            with open(monorepo_env, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:  # Don't override existing values
                        os.environ[key] = val
    except Exception:
        # Best-effort load; ignore errors
        pass


def _load_env(mode: str | None = None) -> tuple[str, str]:
    """Resolve backend base_url and api_key using synth_qwen_v1 pattern.

    Precedence:
    - SYNTH_BACKEND_URL_OVERRIDE=local|dev|prod (preferred)
    - explicit mode arg (local|dev|prod)
    - default prod
    """
    # Load .env files first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Prefer global override if present
    override = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
    if override in {"local", "dev", "prod"}:
        # Try to import from common backend resolver
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from common.backend import resolve_backend_url as _rb
            base = _rb()
        except ImportError:
            # Fallback to hardcoded prod URL
            base = "https://agent-learning.onrender.com/api"
        api_key = os.getenv("SYNTH_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing SYNTH_API_KEY in environment/.env")
        return base.rstrip("/"), api_key

    # Fallback to explicit mode - default to local for automatic setup
    if mode is None:
        mode = os.getenv("SYNTH_MODE", "local").strip().lower()
    if mode == "local":
        base_url = os.getenv("LOCAL_BACKEND_URL", "http://localhost:8000").strip()
        # For local development, try the key from environment, fallback to test key
        api_key = os.getenv("TESTING_LOCAL_SYNTH_API_KEY", os.getenv("SYNTH_API_KEY", "test_local_key")).strip()
        if not base_url:
            raise RuntimeError("Missing LOCAL_BACKEND_URL in environment/.env")
    elif mode == "dev":
        base_url = os.getenv("DEV_BACKEND_URL", "").strip()
        api_key = os.getenv("DEV_SYNTH_API_KEY", "").strip()
        if not base_url or not api_key:
            raise RuntimeError("Missing DEV_BACKEND_URL or DEV_SYNTH_API_KEY in environment/.env")
    else:  # prod
        # Try to import from common backend resolver
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from common.backend import resolve_backend_url as _rb
            base_url = _rb()
        except ImportError:
            # Fallback to hardcoded prod URL
            base_url = "https://agent-learning.onrender.com/api"
        api_key = os.getenv("TESTING_PROD_SYNTH_API_KEY", "").strip() or os.getenv("SYNTH_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing TESTING_PROD_SYNTH_API_KEY/SYNTH_API_KEY in environment/.env")
    return base_url.rstrip("/"), api_key


def _resolve_backend_url(mode: str | None = None) -> str:
    """Resolve backend URL using the same pattern as synth_qwen_v1."""
    base_url, _ = _load_env(mode)
    return base_url


def _parse_args() -> argparse.Namespace:
    # Load .env first so defaults below can see values
    _load_rl_env()
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "dev", "local"], default=None,
                   help="Backend mode/environment (default: env override or prod)")
    # Default to 'local' for localhost development, or use TASK_APP_BASE_URL from env
    p.add_argument("--task-app-url", type=str, default=os.getenv("TASK_APP_BASE_URL", "local").strip())
    p.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "").strip() or "gpt-5-nano")
    # Deprecated: --max-steps is ignored; use --max-steps-each
    p.add_argument("--max-steps", type=int, default=int(os.getenv("RL_MAX_STEPS", "6")))
    # Batch rollout controls (defaults: 10 concurrent, 20 steps each)
    p.add_argument("--num-rollouts", type=int, default=int(os.getenv("RL_NUM_ROLLOUTS", "10")))
    p.add_argument("--max-steps-each", type=int, default=int(os.getenv("RL_MAX_STEPS_EACH", "20")))
    p.add_argument("--use-proxy", action="store_true", help="Route chat via task app /proxy to sanitize GPT-5 payloads")
    p.add_argument("--timeout-seconds", type=int, default=int(os.getenv("RL_TIMEOUT_SECS", "900")), help="Total time to wait for workflow completion")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    # Use mode-based resolution like synth_qwen_v1
    backend_url, api_key = _load_env(args.mode)
    task_app_url = args.task_app_url
    # Optional: source ENVIRONMENT_API_KEY from examples/rl/.env if not present (for diagnostics)
    env_key = os.getenv("ENVIRONMENT_API_KEY", "").strip()
    if not env_key:
        try:
            _env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(_env_path):
                with open(_env_path, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == "ENVIRONMENT_API_KEY":
                            env_key = v.strip().strip('"').strip("'")
                            break
        except Exception:
            pass
    if not api_key:
        print("Missing required input: --api-key (task-app-url defaults to local)")
        return 2

    # Use Modal URL for the task app (it needs to run as a web service)
    if not task_app_url or task_app_url == "local":
        # Resolve from environment; default to local task app for debugging
        endpoint_url = (
            os.getenv("TASK_APP_BASE_URL", "")
            or os.getenv("LOCAL_TASK_APP_URL", "http://localhost:8001")
        ).rstrip("/")
    else:
        endpoint_url = task_app_url.rstrip("/")

    payload = {
        "endpoint_base_url": endpoint_url,
        "model": args.model,
        # Always use max_steps_each to control per-rollout steps
        "max_steps": int(args.max_steps_each),
        # Batch fields consumed by TaskAppOpenAIWorkflow
        "num_rollouts": int(args.num_rollouts),
        "max_steps_each": int(args.max_steps_each),
        "use_proxy": bool(args.use_proxy or ("gpt-5" in (args.model or ""))),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        key_prefix = api_key[:8]
    except Exception:
        key_prefix = "<na>"
    print(
        "Submitting openai-direct:",
        {
            "backend_url": backend_url,
            "task_app_url": task_app_url,
            "model": args.model,
            # Print consolidated step control only
            "max_steps_each": int(args.max_steps_each),
            "num_rollouts": int(args.num_rollouts),
            "use_proxy": bool(args.use_proxy or ("gpt-5" in (args.model or ""))),
            "api_key_prefix": key_prefix,
            "env_key_present": bool(env_key),
            "env_key_prefix": (env_key[:6] if env_key else ""),
        },
    )

    # Submit TaskAppOpenAIWorkflow via backend tests endpoint
    create_url = f"{backend_url.rstrip('/')}/rl/tests/openai/direct"
    # Create job with generous timeouts and simple retries
    _create_timeouts = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=60.0)
    _create_last_exc = None
    for _attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=_create_timeouts) as client:
                resp = await client.post(create_url, json=payload, headers=headers)
            break
        except httpx.ReadTimeout as e:
            _create_last_exc = e
            await asyncio.sleep(2.0 * (1 + _attempt))
        except Exception as e:
            _create_last_exc = e
            await asyncio.sleep(1.0)
    else:
        raise RuntimeError(f"Create request failed after retries: {_create_last_exc}")
    if resp.status_code != 201 and resp.status_code != 200:
        print(f"Create failed: HTTP {resp.status_code} {resp.text[:1200]}")
        return 1
    try:
        js = resp.json()
    except Exception:
        js = {"raw": resp.text[:1200]}
    try:
        import json as _json
        print("create_response:", _json.dumps(js))
    except Exception:
        print("create_response:", js)
    job_id = js.get("job_id") or js.get("id") or ""
    if not job_id:
        # Some implementations may return {ok: True}; treat as success
        print("OK openai-direct:", json.dumps(js)[:400])
        return 0

    # Poll events and status via learning shared endpoints
    status_url = f"{backend_url.rstrip('/')}/learning/jobs/{job_id}"
    events_url = f"{backend_url.rstrip('/')}/learning/jobs/{job_id}/events"
    last_seq = 0
    batch_summary: dict | None = None
    poll_interval = 2.0
    max_iters = int(max(1, args.timeout_seconds / poll_interval))
    for i in range(max_iters):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                ev = await client.get(events_url, params={"since_seq": last_seq, "limit": 200}, headers=headers)
                if ev.status_code == 200:
                    data = ev.json()
                    for e in data.get("events", []):
                        seq = int(e.get("seq") or 0)
                        if seq > last_seq:
                            last_seq = seq
                        et = (e.get("type") or e.get("event_type") or "").lower()
                        # Print full event JSON for maximum visibility
                        try:
                            import json as _json
                            print("event:", _json.dumps(e))
                        except Exception:
                            print("event:", e)

                        # Print rewards when achievements event arrives
                        if et == "test.openai.direct.achievements":
                            try:
                                pdata = e.get("data") or e.get("payload") or {}
                                er = pdata.get("episode_returns")
                                mr = pdata.get("mean_return")
                                if er is not None or mr is not None:
                                    print(f"\nRewards: episode_returns={er} mean_return={mr}")
                            except Exception:
                                pass

                        # If batch summary arrives, capture and pretty-print
                        if et == "test.openai.direct.batch.ok":
                            try:
                                payload_data = e.get("data") or e.get("payload") or {}
                                summary = payload_data.get("summary") or payload_data
                                batch_summary = summary if isinstance(summary, dict) else None
                                if isinstance(batch_summary, dict):
                                    rollouts = batch_summary.get("rollouts") or []
                                    print("\nBatch Summary (achievements and rewards):")
                                    out_lines = ["Batch Summary (achievements and rewards):"]
                                    for r in rollouts:
                                        idx = r.get("index")
                                        ok = r.get("ok")
                                        reward = r.get("reward")
                                        a_last = r.get("achievements_last") or []
                                        a_total = r.get("achievements_total") or []
                                        line = f" - rollout {idx}: ok={ok} reward={reward} unlocked_last={a_last} unlocked_total={a_total}"
                                        print(line)
                                        out_lines.append(line)
                                    # Write summary to file and exit immediately (terminate when done)
                                    try:
                                        out_path = os.path.join(os.path.dirname(__file__), "openai_in_task_rollouts.txt")
                                        with open(out_path, "a", encoding="utf-8") as fh:
                                            fh.write("\n".join(out_lines) + "\n")
                                        print(f"Summary written to {out_path}")
                                    except Exception:
                                        pass
                                    return 0
                            except Exception:
                                pass

                        # Check for workflow completion events
                        if et in ("test.openai.direct.ok", "test.openai.direct.success", "workflow.succeeded", "workflow.completed"):
                            print(f"Workflow completed successfully: {et}")
                            # If we have a captured batch summary, print again at the end in case it arrived earlier
                            if batch_summary and isinstance(batch_summary, dict):
                                rollouts = batch_summary.get("rollouts") or []
                                print("\nFinal Batch Summary:")
                                for r in rollouts:
                                    idx = r.get("index")
                                    ok = r.get("ok")
                                    reward = r.get("reward")
                                    a_last = r.get("achievements_last") or []
                                    a_total = r.get("achievements_total") or []
                                    print(f" - rollout {idx}: ok={ok} reward={reward} unlocked_last={a_last} unlocked_total={a_total}")
                            return 0
                        elif et in ("test.openai.direct.failed", "workflow.failed", "workflow.error"):
                            print(f"Workflow failed: {et}")
                            return 1

                        # On queue_failed, fetch timeline for richer context
                        if "queue_failed" in et:
                            try:
                                timeline_url = f"{backend_url.rstrip('/')}/learning/jobs/{job_id}/timeline"
                                with httpx.Client(timeout=15.0) as c2:
                                    tr = c2.get(timeline_url, params={"limit": 100}, headers=headers)
                                if tr.status_code == 200:
                                    try:
                                        tj = tr.json()
                                        import json as _json
                                        print("timeline:", _json.dumps(tj))
                                    except Exception:
                                        print("timeline_raw:", tr.text[:2000])
                                else:
                                    print(f"timeline_fetch_failed: HTTP {tr.status_code} {tr.text[:400]}")
                            except Exception as te:
                                print(f"timeline_error: {type(te).__name__}: {te}")
                st = await client.get(status_url, headers=headers)
                if st.status_code == 200:
                    sj = st.json()
                    s = str(sj.get("status") or "").lower()
                    if s in ("succeeded", "failed", "canceled", "cancelled"):
                        try:
                            import json as _json
                            print("Final:", _json.dumps(sj))
                        except Exception:
                            print("Final:", sj)
                        return 0 if s == "succeeded" else 1
        except Exception:
            pass
        await asyncio.sleep(poll_interval)

    print(f"Timeout waiting for terminal status after {args.timeout_seconds}s.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
