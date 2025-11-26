from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import click
import httpx

from synth_ai.core.tracing_v3.config import resolve_trace_db_settings
from synth_ai.core.tracing_v3.turso.daemon import start_sqld
from synth_ai.sdk.task.client import TaskAppClient
from synth_ai.sdk.task.contracts import (
    RolloutEnvSpec,
    RolloutMode,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutSafetyConfig,
)
from synth_ai.sdk.task.validators import (
    normalize_inference_url,
    validate_rollout_response_for_rl,
    validate_task_app_url,
)


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = value
    new_query = urlencode(params)
    result = urlunparse(parsed._replace(query=new_query))
    return str(result)


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
        start_sqld(db_path=str(local_db_path), hrana_port=hrana_port, http_port=http_port)
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

    from synth_ai.core.tracing_v3 import config as tracing_config_module
    from synth_ai.core.tracing_v3.storage import config as storage_config_module

    tracing_config_module.CONFIG = tracing_config_module.TursoConfig()  # type: ignore[assignment]
    storage_config_module.STORAGE_CONFIG = storage_config_module.StorageConfig(  # type: ignore[assignment]
        connection_string=os.environ["SYNTH_TRACES_DB"],
        backend=storage_config_module.StorageBackend.TURSO_NATIVE,
    )


def _load_smoke_config(config_path: Path | None) -> dict[str, Any]:
    """Load [smoke] section from TOML config file.
    
    Returns an empty dict if no config file or no [smoke] section.
    """
    if not config_path:
        return {}
    
    try:
        with open(config_path, "rb") as f:
            full_config = tomllib.load(f)
        
        smoke_config = full_config.get("smoke", {})
        
        if smoke_config:
            click.echo(f"[smoke] Loaded configuration from {config_path}", err=True)
            click.echo(f"[smoke] Config keys: {', '.join(smoke_config.keys())}", err=True)
        
        return smoke_config
    except Exception as exc:
        click.echo(f"[smoke] Warning: Failed to load config from {config_path}: {exc}", err=True)
        return {}


def _kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given port."""
    try:
        # Use lsof to find and kill process on port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], timeout=2)
                    click.echo(f"[smoke] Killed existing process {pid} on port {port}", err=True)
                except Exception:
                    pass
            time.sleep(2.0)  # Give OS time to release port
    except Exception as exc:
        click.echo(f"[smoke] Warning: Could not check/kill port {port}: {exc}", err=True)


def _start_task_app_server(
    task_app_name: str,
    port: int,
    env_file: str | None,
    force: bool
) -> tuple[Any, str]:
    """Start a task app server in the background using task-app serve.
    
    Returns (process, url) tuple.
    """
    import subprocess
    import time as time_module
    
    # Build command using task-app serve (for TaskAppConfig-based apps)
    cmd = [
        "nohup",
        "uvx", "synth-ai",
        "task-app", "serve", task_app_name,
        "--port", str(port),
    ]
    
    if env_file:
        cmd.extend(["--env-file", env_file])
    
    if force:
        cmd.append("--force")
    
    # Resolve the synth-ai root directory
    import synth_ai
    synth_ai_root = Path(synth_ai.__file__ or Path(__file__).resolve()).resolve().parent.parent
    
    click.echo(f"[smoke] Starting task app '{task_app_name}' on port {port}...", err=True)
    click.echo(f"[smoke] Command: {' '.join(cmd)}", err=True)
    click.echo(f"[smoke] Working directory: {synth_ai_root}", err=True)
    
    # nohup requires output redirection to a file
    # Open file, start process, then close file handle so process is fully detached
    # Run from synth-ai root so task app discovery works
    nohup_log = Path(synth_ai_root) / "nohup_task_app.out"
    
    # Inherit SYNTH_QUIET environment variable to suppress patch messages
    env = os.environ.copy()
    if os.getenv("SYNTH_QUIET"):
        env["SYNTH_QUIET"] = "1"
    
    with open(nohup_log, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(synth_ai_root),
            env=env,
        )
    # File is closed immediately so process is detached
    
    # Wait for server to be ready
    url = f"http://localhost:{port}"
    click.echo(f"[smoke] Waiting for task app to be ready at {url}...", err=True)
    
    import httpx
    deadline = time.time() + 120.0  # Give it 2 minutes for initial setup
    attempt = 0
    last_log_line = None
    while time.time() < deadline:
        attempt += 1
        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            # Accept both 200 and 400 - 400 means server is up but auth is failing (which is fine for smoke test)
            if resp.status_code in (200, 400):
                click.echo(f"[smoke] Task app ready at {url} (status={resp.status_code})", err=True)
                return proc, url
        except Exception:
            pass
        
        # Show polling progress every 5 seconds with last log line
        if attempt % 10 == 0:
            elapsed = int(time.time() - (deadline - 120.0))
            # Try to read last line from nohup log
            try:
                if nohup_log.exists():
                    with open(nohup_log) as f:
                        lines = f.readlines()
                        if lines:
                            # Get last non-empty line
                            for line in reversed(lines[-10:]):
                                stripped = line.strip()
                                if stripped and stripped != last_log_line:
                                    last_log_line = stripped
                                    # Truncate if too long
                                    if len(stripped) > 80:
                                        stripped = stripped[:77] + "..."
                                    click.echo(f"[smoke] Waiting ({elapsed}s): {stripped}", err=True)
                                    break
                            else:
                                click.echo(f"[smoke] Still waiting for task app... ({elapsed}s elapsed)", err=True)
                        else:
                            click.echo(f"[smoke] Still waiting for task app... ({elapsed}s elapsed)", err=True)
            except Exception:
                click.echo(f"[smoke] Still waiting for task app... ({elapsed}s elapsed)", err=True)
        
        # Check if process died
        if proc.poll() is not None:
            # Build a manual command that the user can copy-paste
            manual_cmd_parts = ["uvx", "synth-ai", "task-app", "serve", task_app_name, "--port", str(port)]
            if env_file:
                manual_cmd_parts.extend(["--env-file", env_file])
            if force:
                manual_cmd_parts.append("--force")
            
            raise click.ClickException(
                f"Task app '{task_app_name}' process exited unexpectedly (code={proc.returncode}). "
                f"Check that the task app name is correct and .env has required keys. "
                f"Try running manually: {' '.join(manual_cmd_parts)}"
            )
        
        time_module.sleep(0.5)
    
    proc.kill()
    raise click.ClickException("Task app failed to start within 120 seconds")


def _start_sqld_server(
    db_path: str,
    hrana_port: int,
    http_port: int
) -> Any:
    """Start sqld server in the background.
    
    Returns the process handle.
    """
    import shutil
    import subprocess
    
    # Check if sqld is available
    sqld_bin = shutil.which("sqld")
    if not sqld_bin:
        click.echo("[smoke] Warning: sqld not found in PATH, skipping auto-start", err=True)
        click.echo("[smoke] Install sqld: brew install sqld", err=True)
        return None
    
    # Ensure db directory exists
    db_path_obj = Path(db_path).expanduser().resolve()
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Kill any existing processes on these ports
    for port in [hrana_port, http_port]:
        _kill_process_on_port(port)
    
    cmd = [
        sqld_bin,
        "--db-path", str(db_path_obj),
        "--hrana-listen-addr", f"127.0.0.1:{hrana_port}",
        "--http-listen-addr", f"127.0.0.1:{http_port}",
    ]
    
    click.echo("[smoke] Starting sqld server...", err=True)
    click.echo(f"[smoke] DB path: {db_path_obj}", err=True)
    click.echo(f"[smoke] Hrana port: {hrana_port}, HTTP port: {http_port}", err=True)
    click.echo(f"[smoke] Command: {' '.join(cmd)}", err=True)
    
    # Redirect to devnull to avoid process dying from pipe buffer issues
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    
    # Wait for server to be ready
    health_url = f"http://127.0.0.1:{http_port}/health"
    click.echo(f"[smoke] Waiting for sqld to be ready at {health_url}...", err=True)
    
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=0.5)
            if resp.status_code == 200:
                click.echo("[smoke] sqld ready", err=True)
                # Set environment variables for tracing
                os.environ["SQLD_DB_PATH"] = str(db_path_obj)
                os.environ["SQLD_HTTP_PORT"] = str(hrana_port)
                os.environ["LIBSQL_URL"] = f"http://127.0.0.1:{http_port}"
                os.environ["SYNTH_TRACES_DB"] = f"http://127.0.0.1:{http_port}"
                return proc
        except Exception:
            pass
        
        # Check if process died
        if proc.poll() is not None:
            click.echo(f"[smoke] Warning: sqld process exited with code {proc.returncode}", err=True)
            return None
        
        time.sleep(0.2)
    
    click.echo("[smoke] Warning: sqld health check timed out, continuing anyway...", err=True)
    return proc

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
        import json

        from fastapi import Body, FastAPI
        from fastapi.responses import JSONResponse

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
                    from examples.task_apps.crafter.task_app.synth_envs_hosted.inference.openai_client import (  # type: ignore[import-untyped]
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
                        if isinstance(message, dict):
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
                # Type-safe extraction of tool call count
                tc = 0
                try:
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices:
                        first_choice = choices[0]
                        if isinstance(first_choice, dict):
                            msg = first_choice.get("message")
                            if isinstance(msg, dict):
                                tool_calls = msg.get("tool_calls")
                                if isinstance(tool_calls, list):
                                    tc = len(tool_calls)
                except Exception:
                    pass
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
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=2.0)
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
            from synth_ai.sdk.api.train.configs.rl import (
                RLConfig as _RLConfig,  # lazy import to avoid heavy deps when unused
            )
            cfg = _RLConfig.from_path(config_path)
        except Exception as exc:
            click.echo(f"Failed to load RL config {config_path}: {exc}", err=True)
            return 2

        # Prefer explicit CLI --url; only use config services.task_url if URL not provided
        try:
            if not task_app_url and cfg.services and getattr(cfg.services, "task_url", None):
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
                # Prefer smoke config model over policy model for smoke tests
                smoke_cfg = getattr(cfg, "smoke", None)
                smoke_model = None
                if smoke_cfg and hasattr(smoke_cfg, "model"):
                    smoke_model = smoke_cfg.model
                if smoke_model:
                    model = str(smoke_model).strip()
                elif cfg.policy:
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
        except Exception:
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
                        click.echo(f"   ops={ops[:10]}{'...' if len(ops) > 10 else ''}")
                        response = await client.rollout(request)
                    except Exception as exc:
                        click.echo(f"Rollout[{i}:{g}] failed: {type(exc).__name__}: {exc}", err=True)
                        import traceback
                        click.echo(f"Traceback: {traceback.format_exc()}", err=True)
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

                        # Extract and display tool calls from v3 trace
                        # 
                        # IMPORTANT: Tool calls are extracted from the structured v3 trace format.
                        # The trace must be requested with return_trace=True for this to work.
                        # 
                        # Trace structure:
                        #   trace.event_history[] - list of events (policy calls, env steps)
                        #     ├─ event.call_records[] - LLM calls made during this event
                        #        ├─ call_record.output_tool_calls[] - tool calls from LLM response
                        #           ├─ tool_call.name - function name (e.g., "interact_many")
                        #           └─ tool_call.arguments_json - JSON string of arguments
                        #
                        # This provides visibility into what actions the policy is taking,
                        # which is critical for debugging RL training issues.
                        tr = response.trace if isinstance(response.trace, dict) else None
                        if tr:
                            event_history = tr.get("event_history", [])
                            tool_call_count = 0
                            
                            # Extract tool calls from event_history call_records
                            if event_history and isinstance(event_history, list):
                                for event in event_history:
                                    if not isinstance(event, dict):
                                        continue
                                    # Policy events contain call_records with LLM interactions
                                    call_records = event.get("call_records")
                                    if call_records and isinstance(call_records, list):
                                        for call_record in call_records:
                                            if isinstance(call_record, dict):
                                                # Extract tool calls from this LLM call
                                                output_tool_calls = call_record.get("output_tool_calls", [])
                                                if output_tool_calls and isinstance(output_tool_calls, list):
                                                    for tc in output_tool_calls:
                                                        if isinstance(tc, dict):
                                                            fn_name = tc.get("name", "unknown")
                                                            fn_args = tc.get("arguments_json", "{}")
                                                            # Display tool call with truncated args for readability
                                                            click.echo(f"  TOOL_CALL[{tool_call_count}]: {fn_name}({fn_args[:100]}{'...' if len(fn_args) > 100 else ''})")
                                                            tool_call_count += 1
                            
                            if tool_call_count > 0:
                                click.echo(f"  ✓ {tool_call_count} tool calls executed")
                            else:
                                # No tool calls found - might indicate:
                                # 1. return_trace=False (trace not requested)
                                # 2. Policy didn't make tool calls (unlikely for most RL tasks)
                                # 3. Trace format mismatch (structure changed)
                                click.echo("  ⚠ No tool calls found in trace")
                        else:
                            click.echo("  ⚠ Trace not available")
                    except Exception as e:
                        click.echo(f"  trace error: {e}", err=True)

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
                with contextlib.suppress(Exception):
                    await mock.stop()
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
        if isinstance(value, int | float):
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
    
    If --config is provided, loads settings from the [smoke] section in the TOML file.
    CLI arguments override TOML values.
    """
    
    # Load [smoke] section from TOML if config is provided
    smoke_config = _load_smoke_config(config)
    
    # Track background processes for cleanup
    background_procs: list[Any] = []
    
    try:
        # Auto-start sqld if configured
        if smoke_config.get("sqld_auto_start"):
            sqld_db_path = smoke_config.get("sqld_db_path", "./traces/local.db")
            sqld_hrana_port = smoke_config.get("sqld_hrana_port", 8080)
            sqld_http_port = smoke_config.get("sqld_http_port", 8081)
            
            sqld_proc = _start_sqld_server(
                db_path=sqld_db_path,
                hrana_port=sqld_hrana_port,
                http_port=sqld_http_port,
            )
            if sqld_proc:
                background_procs.append(("sqld", sqld_proc))
        
        # Auto-start task app if configured
        task_app_override_url = None
        if smoke_config.get("task_app_name"):
            task_app_name = smoke_config["task_app_name"]
            task_app_port = smoke_config.get("task_app_port", 8765)
            task_app_env_file = smoke_config.get("task_app_env_file")
            task_app_force = smoke_config.get("task_app_force", True)
            
            task_app_proc, task_app_url = _start_task_app_server(
                task_app_name=task_app_name,
                port=task_app_port,
                env_file=task_app_env_file,
                force=task_app_force,
            )
            background_procs.append(("task_app", task_app_proc))
            task_app_override_url = task_app_url
            click.echo(f"[smoke] Task app started, will use URL: {task_app_url}", err=True)
    except Exception as exc:
        # Cleanup any processes that did start
        for proc_name, proc in background_procs:
            if proc and proc.poll() is None:
                click.echo(f"[smoke] Cleaning up {proc_name}...", err=True)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        
        click.echo(f"[smoke] ERROR: Auto-start failed: {exc}", err=True)
        raise click.ClickException(f"Auto-start failed: {exc}") from exc
    
    # Apply TOML defaults (CLI args take precedence)
    # Override task_url with auto-started task app URL if applicable
    if task_app_override_url:
        task_app_url = task_app_override_url
    # For string/int args: use TOML value if CLI value matches the default
    ctx = click.get_current_context()
    
    # Helper to check if a CLI param was explicitly provided or is using default
    def use_toml_default(param_name: str, cli_value: Any, toml_key: str) -> Any:
        """Use TOML value if CLI param is at its default, otherwise use CLI value."""
        if not smoke_config or toml_key not in smoke_config:
            return cli_value
        
        param = next((p for p in ctx.command.params if p.name == param_name), None)
        if not param:
            return cli_value
        
        # Check if value was explicitly provided (not default)
        # If it matches the default, use TOML value
        param_default = param.default() if callable(param.default) else param.default
        if cli_value == param_default:
            toml_value = smoke_config[toml_key]
            click.echo(f"[smoke] Using {toml_key}={toml_value} from config", err=True)
            return toml_value
        
        return cli_value
    
    # Apply TOML defaults
    task_app_url = use_toml_default("task_app_url", task_app_url, "task_url")
    env_name = use_toml_default("env_name", env_name, "env_name")
    policy_name = use_toml_default("policy_name", policy_name, "policy_name")
    model = use_toml_default("model", model, "model")
    inference_policy = use_toml_default("inference_policy", inference_policy, "policy")
    inference_url = use_toml_default("inference_url", inference_url, "inference_url")
    max_steps = use_toml_default("max_steps", max_steps, "max_steps")
    return_trace = use_toml_default("return_trace", return_trace, "return_trace")
    use_mock = use_toml_default("use_mock", use_mock, "use_mock")
    mock_backend = use_toml_default("mock_backend", mock_backend, "mock_backend")
    mock_port = use_toml_default("mock_port", mock_port, "mock_port")
    api_key = use_toml_default("api_key", api_key, "api_key")

    # Auto-configure tracing to avoid interactive prompts
    try:
        os.environ.setdefault("CI", "true")
        os.environ.setdefault("SYNTH_TRACING_AUTO_YES", "1")
        # Derive a default traces directory relative to CWD
        traces_dir = os.environ.get("SYNTH_TRACES_DIR")
        if not traces_dir:
            traces_dir = str((Path.cwd() / "traces" / "v3").resolve())
            os.environ["SYNTH_TRACES_DIR"] = traces_dir
        with contextlib.suppress(Exception):
            Path(traces_dir).mkdir(parents=True, exist_ok=True)
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
                from dotenv import find_dotenv as _fd
                from dotenv import load_dotenv as _ld
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
    finally:
        # Cleanup background processes
        for proc_name, proc in background_procs:
            if proc and proc.poll() is None:
                click.echo(f"[smoke] Stopping {proc_name}...", err=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        if background_procs:
            click.echo("[smoke] Background services stopped", err=True)
    
    sys.exit(exit_code)


def register(cli: click.Group) -> None:
    cli.add_command(command)
