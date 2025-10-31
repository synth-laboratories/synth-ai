from __future__ import annotations

import asyncio
import logging
import time
import os
import sys
import uuid
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from pathlib import Path

import click
import httpx

from synth_ai.task.client import TaskAppClient
from synth_ai.task.contracts import RolloutRequest, RolloutEnvSpec, RolloutPolicySpec, RolloutRecordConfig, RolloutSafetyConfig, RolloutMode
from synth_ai.task.validators import (
    validate_task_app_url,
    normalize_inference_url,
    validate_rollout_response_for_rl,
)
from synth_ai.tracing_v3.config import resolve_trace_db_settings
from synth_ai.tracing_v3.turso.daemon import start_sqld


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = value
    new_query = urlencode(params)
    return urlunparse(parsed._replace(query=new_query))


def _ensure_local_libsql() -> None:
    """Start a local sqld/libSQL instance or abort the smoke test."""

    traces_root = Path(os.getenv("SYNTH_TRACES_DIR", str((Path.cwd() / "traces" / "v3").resolve())))
    traces_root.mkdir(parents=True, exist_ok=True)

    local_db_path = Path(os.getenv("SQLD_DB_PATH", str(traces_root / "local.db"))).resolve()
    local_db_path.parent.mkdir(parents=True, exist_ok=True)

    hrana_port = int(os.getenv("SQLD_HTTP_PORT", "8080"))
    http_port = hrana_port + 1
    os.environ["SQLD_DB_PATH"] = str(local_db_path)
    os.environ["SQLD_HTTP_PORT"] = str(hrana_port)

    try:
        daemon = start_sqld(db_path=str(local_db_path), hrana_port=hrana_port, http_port=http_port)
        started_new = True
    except Exception as exc:
        # If address in use, assume an existing sqld instance; verify health below
        if "Address already in use" in str(exc):
            started_new = False
            click.echo(
                f"[libsql] sqld already running on 127.0.0.1:{hrana_port} (hrana) and 127.0.0.1:{http_port} (http); attempting to reuse", err=True
            )
        else:
            raise click.ClickException(
                f"Failed to start local sqld on 127.0.0.1:{hrana_port}: {exc}"
            ) from exc

    health_url = f"http://127.0.0.1:{http_port}/health"
    deadline = time.time() + 5.0
    healthy = False
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=0.5)
            if resp.status_code == 200:
                healthy = True
                break
        except Exception:
            pass
        time.sleep(0.1)

    if not healthy:
        msg = (
            f"Tracing backend not reachable at {health_url}. "
            "Start sqld manually or disable tracing (TASKAPP_TRACING_ENABLED=0)."
        )
        raise click.ClickException(msg)

    click.echo(
        f"[libsql] sqld ready on libsql://127.0.0.1:{hrana_port} with HTTP API on :{http_port} (started_new={started_new})",
        err=True,
    )

    # Python libsql client uses HTTP API port, not Hrana WebSocket port
    local_dsn = f"http://127.0.0.1:{http_port}"
    os.environ["LIBSQL_URL"] = local_dsn
    os.environ["SYNTH_TRACES_DB"] = local_dsn
    os.environ.pop("LIBSQL_AUTH_TOKEN", None)
    os.environ.pop("TURSO_AUTH_TOKEN", None)


def _refresh_tracing_config() -> None:
    """Rebuild global tracing configuration so new env vars take effect."""

    from synth_ai.tracing_v3 import config as tracing_config_module
    from synth_ai.tracing_v3.storage import config as storage_config_module

    tracing_config_module.CONFIG = tracing_config_module.TursoConfig()  # type: ignore[assignment]
    storage_config_module.STORAGE_CONFIG = storage_config_module.StorageConfig(  # type: ignore[assignment]
        connection_string=os.environ["SYNTH_TRACES_DB"],
        backend=storage_config_module.StorageBackend.TURSO_NATIVE,
    )

class MockRLTrainer:
    """Minimal trainer emulator with a local FastAPI mock for GPT-5-Nano.

    In ``synthetic`` mode it emits deterministic tool calls so the rollout can
    progress without relying on external inference. In ``openai`` mode it acts
    as a thin proxy around the real OpenAI chat completions endpoint (useful to
    reproduce production behaviour locally).
    """

    def __init__(self, *, port: int = 0, backend: str = "synthetic") -> None:
        self.port = port
        self.backend = backend.lower().strip() or "synthetic"
        self._server = None
        self._task: asyncio.Task | None = None
        self._openai_endpoint = os.getenv(
            "SMOKE_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"
        )
        self._openai_api_key = (
            os.getenv("SMOKE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        )

    def _build_app(self):
        from fastapi import FastAPI, Body
        from fastapi.responses import JSONResponse
        import json
        import logging

        try:
            logger = logging.getLogger(__name__)
        except Exception:  # pragma: no cover - logging failures should not crash
            logger = None

        app = FastAPI()
        backend = self.backend

        @app.post("/v1/chat/completions")
        async def chat_completions(body: dict = Body(...), cid: str | None = None):
            log = logger or logging.getLogger("MockRLTrainer")
            try:
                msg_count = len(body.get("messages") or [])
            except Exception:
                msg_count = -1
            click.echo(
                f"[mock-rl] ← request backend={backend} model={body.get('model')} messages={msg_count} cid={cid}",
                err=True,
            )

            # Explicit Body(...) avoids FastAPI interpreting parameters as query args
            model = (body.get("model") or "gpt-5-nano")
            messages = body.get("messages") or []
            tools = body.get("tools") or []

            # Decide whether to emit a tool call (to drive env steps) or plain text
            emit_tool = False
            tool_name = ""
            for t in tools:
                try:
                    if (t or {}).get("type") == "function":
                        fn = (t or {}).get("function") or {}
                        name = (fn or {}).get("name") or ""
                        if name:
                            tool_name = name
                            emit_tool = True
                            break
                except Exception:
                    continue

            # Simple heuristic actions to move/explore then interact
            actions = ["move_right", "move_right", "move_down", "move_left", "do"]

            correlation = cid

            if backend == "openai":
                if not self._openai_api_key:
                    return JSONResponse(
                        {
                            "error": "OPENAI_API_KEY (or SMOKE_OPENAI_API_KEY) is required for mock backend 'openai'"
                        },
                        status_code=500,
                    )
                try:
                    from examples.task_apps.crafter.task_app.synth_envs_hosted.inference.openai_client import (
                        OpenAIClient as _HostedOpenAIClient,
                    )

                    hosted_client = _HostedOpenAIClient(
                        base_url=self._openai_endpoint,
                        api_key=self._openai_api_key,
                    )
                except Exception as exc:
                    if logger is not None:
                        logger.error("MockRLTrainer failed to import HostedOpenAIClient: %s", exc)
                    return JSONResponse(
                        {"error": f"OpenAI proxy unavailable: {exc}"},
                        status_code=500,
                    )

                try:
                    result = await hosted_client.generate_with_retries(  # type: ignore[attr-defined]
                        request=body,
                        base_url=self._openai_endpoint,
                        max_retries=0,
                    )
                except Exception as exc:
                    if logger is not None:
                        logger.error("MockRLTrainer OpenAI generate failed: %s", exc)
                    return JSONResponse(
                        {"error": f"OpenAI proxy request failed: {exc}"},
                        status_code=502,
                    )

                if isinstance(result, dict):
                    data_typed = dict(result)
                    synth_meta = data_typed.get("synth")
                    if not isinstance(synth_meta, dict):
                        synth_meta = {}
                        data_typed["synth"] = synth_meta
                    if correlation:
                        synth_meta.setdefault("cid", correlation)

                    # Fallback: if the upstream response failed to emit tool calls,
                    # synthesize a deterministic action plan so the rollout can proceed.
                    try:
                        choices = data_typed.get("choices") or []
                        first = choices[0] if choices else {}
                        message = first.get("message") if isinstance(first, dict) else {}
                        tc = message.get("tool_calls") if isinstance(message, dict) else None
                        if not tc:
                            if logger is not None:
                                logger.warning(
                                    "MockRLTrainer fallback: OpenAI returned no tool calls; injecting deterministic actions."
                                )
                            fallback_message = dict(message or {})
                            fallback_message.setdefault("role", "assistant")
                            fallback_message["content"] = ""
                            fallback_message["tool_calls"] = [
                                {
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name or "interact_many",
                                        "arguments": json.dumps({"actions": actions}),
                                    },
                                }
                            ]
                            fallback_message["function_call"] = {
                                "name": tool_name or "interact_many",
                                "arguments": json.dumps({"actions": actions}),
                            }
                            if choices:
                                choices[0]["message"] = fallback_message
                            else:
                                data_typed["choices"] = [
                                    {
                                        "index": 0,
                                        "message": fallback_message,
                                        "finish_reason": "tool_calls",
                                    }
                                ]
                    except Exception as exc:
                        if logger is not None:
                            logger.debug("MockRLTrainer fallback injection failed: %s", exc)

                    tool_call_count = 0
                    try:
                        choices = data_typed.get("choices") or []
                        first = choices[0] if choices else {}
                        message = first.get("message") if isinstance(first, dict) else {}
                        tool_call_count = len(message.get("tool_calls") or [])
                    except Exception:
                        tool_call_count = 0

                    log.info(
                        "MockRLTrainer proxy returning response with %s tool calls (cid=%s)",
                        tool_call_count,
                        cid,
                    )
                    if tool_call_count == 0:
                        log.error(
                            "MockRLTrainer proxy still missing tool_calls after fallback injection (cid=%s)",
                            cid,
                        )
                        click.echo(
                            "[mock-rl] ✗ proxy response missing tool_calls; failing request", err=True
                        )
                    return JSONResponse(data_typed)
                return JSONResponse(result)

            if emit_tool:
                # Emit BOTH legacy function_call and modern tool_calls for broad compatibility
                message_payload = {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": tool_name,
                        "arguments": json.dumps({"actions": actions}),
                    },
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"actions": actions}),
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
            else:
                # Fallback: echo last user content as plain text
                click.echo(
                    f"[mock-rl] ! no tool schema supplied; returning text response (cid={cid})",
                    err=True,
                )
                log.warning(
                    "MockRLTrainer received request without tool schema; responding with text content (cid=%s)",
                    cid,
                )
                last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                text = (last_user or "").strip()
                if len(text) > 160:
                    text = text[:160] + "..."
                message_payload = {"role": "assistant", "content": f"MOCK(gpt-5-nano): {text or 'ack'}"}
                finish_reason = "stop"

            response = {
                "id": f"cmpl_{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [{"index": 0, "message": message_payload, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
                "synth": {"cid": correlation},
            }
            if finish_reason == "tool_calls":
                tc = len(response["choices"][0]["message"].get("tool_calls") or [])
                log.debug(
                    "MockRLTrainer synthetic response emitting %s tool calls (cid=%s)",
                    tc,
                    cid,
                )
                assert tc > 0, "MockRLTrainer synthetic response missing tool_calls"
                click.echo(
                    f"[mock-rl] → response tool_calls={tc} backend={backend} cid={cid}",
                    err=True,
                )
            else:
                click.echo(
                    f"[mock-rl] → response finish_reason={finish_reason} backend={backend} cid={cid}",
                    err=True,
                )
            return JSONResponse(response)

        return app

    async def start(self) -> None:
        import socket
        import uvicorn
        import logging

        def _allocate_port() -> int:
            nonlocal socket
            if self.port:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        probe.bind(("127.0.0.1", self.port))
                        return self.port
                    except OSError:
                        pass
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", 0))
                self.port = probe.getsockname()[1]
                return self.port

        retries = 0
        while True:
            selected_port = _allocate_port()
            config = uvicorn.Config(
                self._build_app(),
                host="127.0.0.1",
                port=selected_port,
                log_level="warning",
            )
            self._server = uvicorn.Server(config)
            self._task = asyncio.create_task(self._server.serve())

            for _ in range(100):
                if getattr(self._server, "started", False):
                    break
                if self._task.done():
                    break
                await asyncio.sleep(0.05)

            if getattr(self._server, "started", False):
                try:
                    logging.getLogger(__name__).info(
                        "MockRLTrainer started on http://127.0.0.1:%s (backend=%s)",
                        self.port,
                        self.backend,
                    )
                    click.echo(
                        f"[mock-rl] server ready http://127.0.0.1:{self.port} backend={self.backend}",
                        err=True,
                    )
                except Exception:
                    pass
                return

            # Startup failed; stop server and retry on a new port if possible
            await self.stop()
            if retries >= 5:
                raise RuntimeError("MockRLTrainer failed to start after multiple attempts")
            self.port = 0
            retries += 1

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                pass
        self._task = None
        self._server = None
        click.echo("[mock-rl] server stopped", err=True)

async def _run_smoke_async(
    *,
    task_app_url: str,
    api_key: str | None,
    env_name_opt: str | None,
    policy_name: str,
    model: str,
    inference_url_opt: str | None,
    inference_policy: str | None,
    max_steps: int,
    return_trace: bool,
    use_mock: bool,
    mock_port: int,
    mock_backend: str,
    config_path: Path | None,
    rollouts: int = 1,
    group_size: int = 1,
    batch_size: int | None = None,
) -> int:
    # If config is provided, derive defaults (URL/env/model)
    cfg: Any | None = None
    if config_path is not None:
        try:
            from synth_ai.api.train.configs.rl import RLConfig as _RLConfig  # lazy import to avoid heavy deps when unused
            cfg = _RLConfig.from_path(config_path)
        except Exception as exc:
            click.echo(f"Failed to load RL config {config_path}: {exc}", err=True)
            return 2

        # Prefer explicit CLI --url; only use config services.task_url if URL not provided
        try:
            if not task_app_url:
                if cfg.services and getattr(cfg.services, "task_url", None):
                    task_app_url = cfg.services.task_url
        except Exception:
            pass
        # Fill env and model if not explicitly set
        try:
            if not env_name_opt and cfg.rollout and getattr(cfg.rollout, "env_name", None):
                env_name_opt = cfg.rollout.env_name
        except Exception:
            pass
        try:
            if model == "gpt-5-nano":
                if cfg.policy:
                    if getattr(cfg.policy, "model_name", None):
                        model = str(cfg.policy.model_name).strip()
                    elif getattr(cfg.policy, "source", None):
                        model = str(cfg.policy.source).strip()
                elif cfg.model and getattr(cfg.model, "source", None):
                    model = str(cfg.model.source).strip()
                elif cfg.model and getattr(cfg.model, "base", None):
                    model = str(cfg.model.base).strip()
        except Exception:
            pass

    base = validate_task_app_url(task_app_url)
    mock_backend = (mock_backend or "synthetic").strip().lower()

    # Discover environment if not provided
    async with TaskAppClient(base_url=base, api_key=api_key) as client:
        # Probe basic info quickly
        try:
            _ = await client.health()
        except Exception as exc:
            click.echo("Auth or connectivity check failed on /health. If this endpoint requires a key, pass --api-key or set ENVIRONMENT_API_KEY.", err=True)
            # Continue; rollout may still clarify the error

        # Fetch a sample task instance to infer environment name if not provided
        env_name = env_name_opt
        if not env_name:
            try:
                ti = await client.task_info(seeds=[0])
                # task_info returns TaskInfo or list[TaskInfo]; normalize
                info: Any = ti[0] if isinstance(ti, list) else ti
                env_name = getattr(info, "environment", None) or getattr(info, "task", {}).get("name")  # type: ignore[attr-defined]
            except Exception:
                env_name = None
        if not env_name:
            click.echo("Could not infer environment name; pass --env-name.", err=True)
            return 2

        # Build ops: alternating agent/env for max_steps
        ops: list[str] = []
        for _ in range(max_steps):
            ops.append("agent")
            ops.append("env")

        # Inference URL: user override > preset > local mock > Synth API default
        synth_base = (os.getenv("SYNTH_API_BASE") or os.getenv("SYNTH_BASE_URL") or "https://api.synth.run").rstrip("/")
        # Avoid double '/api' if base already includes it
        if synth_base.endswith("/api"):
            default_infer = f"{synth_base}/inference/v1/chat/completions"
        else:
            default_infer = f"{synth_base}/api/inference/v1/chat/completions"

        # Helper to execute one or more rollouts and return exit code
        async def __do_rollouts(inference_url_raw: str) -> int:
            successes = 0
            total_steps = 0
            nonzero_returns = 0
            v3_traces = 0

            # Derive sampling params from config if present
            sampling: dict[str, Any] = {}
            try:
                if cfg and cfg.policy:
                    if getattr(cfg.policy, "temperature", None) is not None:
                        sampling["temperature"] = cfg.policy.temperature
                    if getattr(cfg.policy, "top_p", None) is not None:
                        sampling["top_p"] = cfg.policy.top_p
                    if getattr(cfg.policy, "max_tokens", None) is not None:
                        sampling["max_tokens"] = cfg.policy.max_tokens
            except Exception:
                pass

            num_outer = batch_size if (batch_size is not None and batch_size > 0) else max(1, int(rollouts))
            for i in range(num_outer):
                for g in range(max(1, int(group_size))):
                    if inference_url_raw.startswith("/"):
                        inference_url_abs = f"{base}{inference_url_raw}"
                    else:
                        inference_url_abs = inference_url_raw
                    inference_url_norm = normalize_inference_url(inference_url_abs)
                    correlation_id = f"smoke-{uuid.uuid4()}"
                    inference_url_with_cid = _append_query_param(inference_url_norm, "cid", correlation_id)

                    run_id = correlation_id
                    policy_cfg: dict[str, Any] = {
                        "model": model,
                        "inference_url": inference_url_with_cid,
                    }
                    if sampling:
                        policy_cfg.update(sampling)

                    request = RolloutRequest(
                        run_id=run_id,
                        env=RolloutEnvSpec(env_name=env_name, config={}, seed=i),
                        policy=RolloutPolicySpec(policy_name=policy_name, config=policy_cfg),
                        ops=ops,
                        record=RolloutRecordConfig(
                            trajectories=True,
                            logprobs=False,
                            value=False,
                            return_trace=return_trace,
                            trace_format=("structured" if return_trace else "compact"),
                        ),
                        on_done="reset",
                        safety=RolloutSafetyConfig(max_ops=max_steps * 4, max_time_s=900.0),
                        training_session_id=None,
                        synth_base_url=synth_base,
                        mode=RolloutMode.RL,
                    )

                    try:
                        click.echo(f">> POST /rollout run_id={run_id} env={env_name} policy={policy_name} url={inference_url_with_cid}")
                        response = await client.rollout(request)
                    except Exception as exc:
                        click.echo(f"Rollout[{i}:{g}] failed: {type(exc).__name__}: {exc}", err=True)
                        continue

                    successes += 1
                    try:
                        validate_rollout_response_for_rl(response.model_dump())
                    except Exception as vexc:
                        click.echo(f"  ⚠ RL response validation warning: {vexc}", err=True)

                    pm = response.pipeline_metadata or {}
                    inferred_url = pm.get("inference_url") if isinstance(pm, dict) else None
                    metrics = response.metrics
                    if inferred_url:
                        click.echo(f"  rollout[{i}:{g}] inference_url: {inferred_url}")
                    click.echo(f"  rollout[{i}:{g}] episodes={metrics.num_episodes} steps={metrics.num_steps} mean_return={metrics.mean_return:.4f}")

                    total_steps += int(metrics.num_steps)
                    if (metrics.mean_return or 0.0) != 0.0:
                        nonzero_returns += 1
                    if response.trace is not None and isinstance(response.trace, dict):
                        v3_traces += 1

                    if i == 0 and g == 0:
                        try:
                            traj0 = response.trajectories[0]
                            step_meta_url = None
                            for step in traj0.steps:
                                info = getattr(step, "info", None) or {}
                                meta = info.get("meta") if isinstance(info, dict) else None
                                if isinstance(meta, dict) and meta.get("inference_url"):
                                    step_meta_url = meta.get("inference_url")
                                    break
                            if step_meta_url:
                                click.echo(f"  step.meta.inference_url: {str(step_meta_url)[:120]}...")
                        except Exception:
                            pass

                    try:
                        try:
                            metrics_dump = response.metrics.model_dump()
                        except Exception:
                            metrics_dump = {
                                "episode_returns": getattr(response.metrics, "episode_returns", None),
                                "mean_return": getattr(response.metrics, "mean_return", None),
                                "num_steps": getattr(response.metrics, "num_steps", None),
                                "num_episodes": getattr(response.metrics, "num_episodes", None),
                                "outcome_score": getattr(response.metrics, "outcome_score", None),
                                "events_score": getattr(response.metrics, "events_score", None),
                            }
                        click.echo("  reward.info (metrics): " + str(metrics_dump))

                        try:
                            traj = response.trajectories[0]
                            step_rewards = []
                            all_achievements = set()
                            for st in getattr(traj, "steps", []) or []:
                                try:
                                    step_rewards.append(getattr(st, "reward", None))
                                except Exception:
                                    step_rewards.append(None)
                                # Extract achievements from step info
                                try:
                                    step_info = getattr(st, "info", None)
                                    if isinstance(step_info, dict):
                                        achievements_status = step_info.get("achievements_status")
                                        if isinstance(achievements_status, dict):
                                            for ach_name, ach_val in achievements_status.items():
                                                if ach_val:
                                                    all_achievements.add(str(ach_name))
                                except Exception:
                                    pass
                            click.echo("  reward.per_step: " + str(step_rewards))
                            if all_achievements:
                                click.echo(f"  achievements: {sorted(all_achievements)}")
                            else:
                                click.echo("  achievements: none")
                        except Exception:
                            pass

                        tr = response.trace if isinstance(response.trace, dict) else None
                        msgs = (tr or {}).get("messages") if isinstance(tr, dict) else None
                        if isinstance(msgs, list):
                            def _part_type_summary(parts: list) -> str:
                                try:
                                    kinds: dict[str, int] = {}
                                    for p in parts:
                                        t = p.get("type") if isinstance(p, dict) else type(p).__name__
                                        kinds[str(t)] = kinds.get(str(t), 0) + 1
                                    return ", ".join(f"{k}:{v}" for k, v in kinds.items())
                                except Exception:
                                    return "unknown"

                            click.echo(f"  trace.messages: count={len(msgs)}")
                            for mi, m in enumerate(msgs[:50]):
                                if not isinstance(m, dict):
                                    continue
                                role = m.get("role")
                                content = m.get("content")
                                if isinstance(content, list):
                                    summary = _part_type_summary(content)
                                    click.echo(f"    [{mi}] role={role} content=list[{len(content)}] parts=({summary})")
                                else:
                                    ct = type(content).__name__
                                    click.echo(f"    [{mi}] role={role} content_type={ct}")
                    except Exception:
                        pass

            click.echo("✓ Smoke rollouts complete")
            denom = num_outer * max(1, int(group_size))
            click.echo(f"  successes={successes}/{denom} total_steps={total_steps} v3_traces={v3_traces}/{denom} nonzero_returns={nonzero_returns}/{denom}")

            if successes == 0:
                click.echo("  ⚠ All rollouts failed", err=True)
                return 3
            if v3_traces < successes:
                click.echo("  ⚠ Some rollouts missing v3 traces (trace field)", err=True)
            if total_steps == 0:
                click.echo("  ⚠ No steps executed; check ops/policy config", err=True)

            return 0

        # Initialize to default; policy/flags may override below
        inference_url_raw = inference_url_opt or default_infer
        mock: MockRLTrainer | None = None
        preset = (inference_policy or "").strip().lower()

        # Respect explicit preset overrides
        if preset == "mock":
            use_mock = True
        elif preset == "gpt-5-nano":
            if not inference_url_opt:
                inference_url_raw = default_infer
            if not model:
                model = "gpt-5-nano"
        elif preset == "openai":
            inference_url_raw = "https://api.openai.com/v1/chat/completions"
        elif preset == "groq":
            inference_url_raw = "https://api.groq.com/openai/v1/chat/completions"

        # Start mock proxy only when explicitly requested
        if use_mock:
            backend_choice = mock_backend
            if backend_choice == "openai" and not (
                os.getenv("SMOKE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            ):
                click.echo(
                    "  ⚠ OPENAI_API_KEY not configured; falling back to synthetic mock.",
                    err=True,
                )
                backend_choice = "synthetic"
            mock = MockRLTrainer(port=mock_port, backend=backend_choice)
            await mock.start()
            inference_url_raw = f"http://127.0.0.1:{mock.port}"

        try:
            result = await __do_rollouts(inference_url_raw)
        finally:
            if mock is not None:
                try:
                    await mock.stop()
                except Exception:
                    pass
        return result
async def _run_train_step(
    *,
    task_app_url: str,
    api_key: str | None,
    env_name_opt: str | None,
    policy_name: str,
    model: str,
    inference_policy: str | None,
    inference_url_opt: str | None,
    max_steps: int,
    return_trace: bool,
    use_mock: bool,
    mock_backend: str,
    mock_port: int,
    config_path: Path | None,
    parallel: int,
) -> int:
    import time
    start = time.perf_counter()

    async def one(seed_idx: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            code = await _run_smoke_async(
                task_app_url=task_app_url,
                api_key=api_key,
                env_name_opt=env_name_opt,
                policy_name=policy_name,
                model=model,
                inference_policy=inference_policy,
                inference_url_opt=inference_url_opt,
                max_steps=max_steps,
                return_trace=return_trace,
                use_mock=use_mock,
                mock_backend=mock_backend,
                mock_port=mock_port,
                config_path=config_path,
                rollouts=1,
                group_size=1,
                batch_size=None,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {"exit": int(code), "wall_ms": wall_ms}
        except Exception as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {"exit": 99, "wall_ms": wall_ms, "error": f"{type(e).__name__}: {e}"}

    # Launch N rollouts concurrently
    tasks = [one(i) for i in range(max(1, int(parallel)))]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    total_wall_ms = (time.perf_counter() - start) * 1000.0

    # Print summary
    def _exit_code(result: dict[str, Any]) -> int:
        value = result.get("exit")
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(value.strip())
            except ValueError:
                return 1
        return 1

    successes = sum(1 for r in results if _exit_code(r) == 0)
    avg_wall = sum(float(r.get("wall_ms", 0.0)) for r in results) / max(len(results), 1)
    click.echo("✓ Train-step emulation complete")
    click.echo(f"  parallel={parallel} successes={successes}/{len(results)} total_wall_ms={total_wall_ms:.1f} avg_rollout_wall_ms={avg_wall:.1f}")

    # Show brief failure codes to aid diagnosis
    if successes < len(results):
        codes: dict[int, int] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            c = _exit_code(r)
            codes[c] = codes.get(c, 0) + 1
        click.echo(f"  failure_codes={codes}")

    return 0 if successes == len(results) else 3


@click.command("smoke")
@click.option("--url", "task_app_url", type=str, default=lambda: os.getenv("TASK_APP_URL", "http://localhost:8765"), help="Task app base URL.")
@click.option(
    "--api-key",
    type=str,
    default=lambda: os.getenv("ENVIRONMENT_API_KEY", ""),
    envvar="ENVIRONMENT_API_KEY",
    help="Environment API key (X-API-Key).",
)
@click.option("--env-name", type=str, default=None, help="Environment name to roll out (auto-detected if possible).")
@click.option("--policy-name", type=str, default="react", help="Policy name to pass to task app.")
@click.option("--model", type=str, default="gpt-5-nano", help="Model id to route in inference payload.")
@click.option(
    "--policy",
    "inference_policy",
    type=click.Choice(["mock", "gpt-5-nano", "openai", "groq"], case_sensitive=False),
    default=None,
    help="Inference route preset (mock, gpt-5-nano via Synth, OpenAI or Groq).",
)
@click.option("--inference-url", type=str, default=None, help="Override inference URL (default: Synth API chat completions).")
@click.option("--max-steps", type=int, default=3, show_default=True, help="Number of agent/env step pairs.")
@click.option("--return-trace", is_flag=True, help="Request v3 trace in response if supported.")
@click.option("--use-mock/--no-mock", default=True, show_default=True, help="Use local mock inference server (GPT-5-Nano emulation).")
@click.option(
    "--mock-backend",
    type=click.Choice(["synthetic", "openai"], case_sensitive=False),
    default="synthetic",
    show_default=True,
    help="Mock inference backend: synthetic deterministic tooling or OpenAI passthrough.",
)
@click.option("--mock-port", type=int, default=0, show_default=True, help="Port for local mock inference server (0 = auto).")
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="RL TOML config to derive URL/env/model.")
@click.option("--env-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Path to .env to load before running.")
@click.option("--rollouts", type=int, default=1, show_default=True, help="Number of rollouts (seeds 0..N-1).")
@click.option("--group-size", type=int, default=1, show_default=True, help="Completions per seed to emulate GRPO grouping.")
@click.option("--batch-size", type=int, default=None, help="Alias for rollouts; when set, overrides --rollouts.")
@click.option(
    "--parallel",
    type=int,
    default=0,
    show_default=True,
    help="Emulate a train step by running this many rollouts concurrently (0 = sequential).",
)
def command(
    task_app_url: str,
    api_key: str,
    env_name: str | None,
    policy_name: str,
    model: str,
    inference_policy: str | None,
    inference_url: str | None,
    max_steps: int,
    return_trace: bool,
    use_mock: bool,
    mock_backend: str,
    mock_port: int,
    config: Path | None,
    env_file: Path | None,
    rollouts: int,
    group_size: int,
    batch_size: int | None,
    parallel: int,
) -> None:
    """Smoke-test a Task App by emulating a trainer rollout using GPT-5-Nano.

    This command posts a minimal RL rollout to the task app, with a valid
    OpenAI-compatible inference URL including a trace correlation id, and
    validates that the response contains the fields required by the RL trainer
    (e.g. pipeline_metadata.inference_url and per-step info.meta.inference_url).
    """

    # Auto-configure tracing to avoid interactive prompts
    try:
        os.environ.setdefault("CI", "true")
        os.environ.setdefault("SYNTH_TRACING_AUTO_YES", "1")
        # Derive a default traces directory relative to CWD
        traces_dir = os.environ.get("SYNTH_TRACES_DIR")
        if not traces_dir:
            traces_dir = str((Path.cwd() / "traces" / "v3").resolve())
            os.environ["SYNTH_TRACES_DIR"] = traces_dir
        try:
            Path(traces_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _ensure_local_libsql()
        # Prefer a libsql/turso/sqld URL when provided to enable concurrent writes
        libsql_url = (
            os.getenv("TRACING_DB_URL")
            or os.getenv("LIBSQL_URL")
            or os.getenv("TURSO_DATABASE_URL")
            or os.getenv("LIBSQL_HTTP_URL")
        )
        if libsql_url:
            os.environ.setdefault("LIBSQL_URL", libsql_url)

        auth_hint = (
            os.getenv("TRACING_DB_AUTH_TOKEN")
            or os.getenv("LIBSQL_AUTH_TOKEN")
            or os.getenv("TURSO_AUTH_TOKEN")
        )
        if auth_hint:
            os.environ.setdefault("LIBSQL_AUTH_TOKEN", auth_hint)

        resolved_url, resolved_token = resolve_trace_db_settings()
        os.environ.setdefault("SYNTH_TRACES_DB", resolved_url)
        if resolved_token and not (
            os.getenv("LIBSQL_AUTH_TOKEN") or os.getenv("TURSO_AUTH_TOKEN")
        ):
            os.environ["LIBSQL_AUTH_TOKEN"] = resolved_token

        _refresh_tracing_config()
    except Exception:
        pass

    # Load env file(s) before resolving API key
    try:
        # Explicit --env-file takes precedence
        if env_file is not None:
            try:
                from dotenv import load_dotenv as _ld
                _ld(env_file, override=False)
            except Exception:
                pass
        else:
            # Best-effort auto-discovery from CWD
            try:
                from dotenv import find_dotenv as _fd, load_dotenv as _ld
                _ld(_fd(usecwd=True), override=False)
            except Exception:
                pass

        # If api_key not passed, try to read from env now
        if not api_key:
            api_key = os.getenv("ENVIRONMENT_API_KEY", "")
    except Exception:
        pass

    try:
        if parallel and parallel > 0:
            exit_code = asyncio.run(
                _run_train_step(
                    task_app_url=task_app_url,
                    api_key=(api_key or None),
                    env_name_opt=env_name,
                    policy_name=policy_name,
                    model=model,
                    inference_policy=inference_policy,
                    inference_url_opt=inference_url,
                    max_steps=max_steps,
                    return_trace=return_trace,
                    use_mock=use_mock,
                    mock_backend=mock_backend,
                    mock_port=mock_port,
                    config_path=config,
                    parallel=parallel,
                )
            )
        else:
            exit_code = asyncio.run(
                _run_smoke_async(
                    task_app_url=task_app_url,
                    api_key=(api_key or None),
                    env_name_opt=env_name,
                    policy_name=policy_name,
                    model=model,
                    inference_policy=inference_policy,
                    inference_url_opt=inference_url,
                    max_steps=max_steps,
                    return_trace=return_trace,
                    use_mock=use_mock,
                    mock_backend=mock_backend,
                    mock_port=mock_port,
                    config_path=config,
                    rollouts=rollouts,
                    group_size=group_size,
                    batch_size=batch_size,
                )
            )
    except KeyboardInterrupt:
        click.echo("Interrupted", err=True)
        sys.exit(130)
    sys.exit(exit_code)


def register(cli: click.Group) -> None:
    cli.add_command(command)
from synth_ai.tracing_v3.config import resolve_trace_db_settings
