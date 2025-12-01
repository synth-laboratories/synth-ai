import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import uvicorn
from starlette.types import ASGIApp

from synth_ai.core.apps.common import get_asgi_app, load_module
from synth_ai.core.cfgs import LocalDeployCfg
from synth_ai.core.paths import REPO_ROOT, configure_import_paths
from synth_ai.core.telemetry import log_error, log_info

_THREADS: dict[int, threading.Thread] = {}
_PROCESSES: dict[int, subprocess.Popen[str]] = {}


def serve_app_uvicorn(
    app: ASGIApp,
    host: str,
    port: int,
) -> None:
    ctx: dict[str, Any] = {"host": host, "port": port}
    log_info("starting uvicorn server", ctx=ctx)
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        log_info("uvicorn server exited cleanly", ctx=ctx)
    except Exception as exc:
        ctx["error"] = type(exc).__name__
        log_error("uvicorn server failed", ctx=ctx)
        raise RuntimeError(f"Failed to serve app with Uvicorn: {exc}") from exc
    

def serve_app_uvicorn_background(
    app: ASGIApp,
    host: str,
    port: int,
    daemon: bool = True
) -> None:
    thread = threading.Thread(
        target=serve_app_uvicorn,
        args={app, host, port},
        name=f"synth-uvicorn-{port}",
        daemon=daemon
    )
    thread.start()


def deploy_app_uvicorn(cfg: LocalDeployCfg) -> str | None:
    ctx: dict[str, Any] = {
        "task_app_path": str(cfg.task_app_path),
        "host": cfg.host,
        "port": cfg.port,
        "trace": cfg.trace,
    }
    log_info("deploy_app_uvicorn invoked", ctx=ctx)
    try:
        os.environ["ENVIRONMENT_API_KEY"] = cfg.env_api_key
        if cfg.trace:
            os.environ["TASKAPP_TRACING_ENABLED"] = "1"
        else:
            os.environ.pop("TASKAPP_TRACING_ENABLED", None)

        configure_import_paths(cfg.task_app_path, REPO_ROOT)
        module = load_module(
            cfg.task_app_path,
            f"_synth_local_task_app_{cfg.task_app_path.stem}"
        )
        log_info("task app module loaded", ctx=ctx)
        app = get_asgi_app(module)

        msg = f"Serving task app at http://{'127.0.0.1' if cfg.host in {'0.0.0.0', '::'} else cfg.host}:{cfg.port}"

        # Try to extract app_id from module or app
        app_id: str | None = None
        try:
            # Try to get app_id from module's config factory or app state
            if hasattr(module, "build_config"):
                config = module.build_config()  # type: ignore[call-arg]
                if hasattr(config, "app_id"):
                    app_id = config.app_id
            elif hasattr(app, "state") and hasattr(app.state, "task_app_config"):
                config = app.state.task_app_config
                if hasattr(config, "app_id"):
                    app_id = str(config.app_id) if config.app_id is not None else None  # type: ignore[assignment]
        except Exception:
            pass

        # Record local service will happen after process starts (to get correct PID)

        if os.environ.get("CTX") == "mcp":
            thread = threading.Thread(
                target=serve_app_uvicorn,
                args=(app, cfg.host, cfg.port),
                name=f"synth-uvicorn-{cfg.port}",
                daemon=True
            )
            thread.start()
            _THREADS[cfg.port] = thread
            ctx["mode"] = "mcp"
            log_info("uvicorn server running in background thread", ctx=ctx)
            return f"[deploy_local] {msg}"

        # Use nohup to run in background
        print(msg)
        print("[deploy_local] Starting server in background with nohup...")

        # Create a wrapper script to run uvicorn
        import tempfile
        task_app_path_str = str(cfg.task_app_path)
        repo_root_str = str(REPO_ROOT)
        task_app_parent_str = str(cfg.task_app_path.parent)
        module_name = f"_synth_local_task_app_{cfg.task_app_path.stem}"

        wrapper_content = f'''#!/usr/bin/env python3
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, {repr(repo_root_str)})
sys.path.insert(0, {repr(task_app_parent_str)})

# Configure import paths
from synth_ai.core.paths import configure_import_paths
configure_import_paths(Path({repr(task_app_path_str)}), Path({repr(repo_root_str)}))

# Load module and app
from synth_ai.core.apps.common import get_asgi_app, load_module

module = load_module(
    Path({repr(task_app_path_str)}),
    {repr(module_name)}
)
app = get_asgi_app(module)

# Set environment
import os
os.environ["ENVIRONMENT_API_KEY"] = {repr(cfg.env_api_key)}
'''
        if cfg.trace:
            wrapper_content += 'os.environ["TASKAPP_TRACING_ENABLED"] = "1"\n'
        else:
            wrapper_content += 'os.environ.pop("TASKAPP_TRACING_ENABLED", None)\n'

        wrapper_content += f'''
# Run uvicorn
import uvicorn
uvicorn.run(app, host={repr(cfg.host)}, port={cfg.port}, reload=False, log_level="info")
'''

        # Write wrapper script
        wrapper_file = Path(tempfile.gettempdir()) / f"synth_uvicorn_{cfg.port}.py"
        wrapper_file.write_text(wrapper_content)
        wrapper_file.chmod(0o755)

        # Create log file
        log_file = Path(tempfile.gettempdir()) / f"synth_uvicorn_{cfg.port}.log"

        # Build command with nohup
        cmd = [
            "nohup",
            sys.executable,
            str(wrapper_file),
        ]

        # Start process with nohup
        try:
            with log_file.open("w") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(REPO_ROOT),
                    env=os.environ.copy(),
                )
            _PROCESSES[cfg.port] = proc

            # Record local service with correct PID
            try:
                from synth_ai.cli.lib.tunnel_records import record_service
                local_url = f"http://{'127.0.0.1' if cfg.host in {'0.0.0.0', '::'} else cfg.host}:{cfg.port}"
                record_service(
                    url=local_url,
                    port=cfg.port,
                    service_type="local",
                    local_host=cfg.host if cfg.host not in ('0.0.0.0', '::') else '127.0.0.1',
                    task_app_path=str(cfg.task_app_path) if cfg.task_app_path else None,
                    app_id=app_id,
                    pid=proc.pid,
                )
            except Exception:
                pass  # Fail silently - records are optional

            print(f"[deploy_local] Server started (PID: {proc.pid})")
            print(f"[deploy_local] Logs: {log_file}")
            print(f"[deploy_local] URL: {msg}")
            ctx["pid"] = proc.pid
            ctx["log_file"] = str(log_file)
            log_info("uvicorn server started with nohup", ctx=ctx)
            return f"[deploy_local] {msg} (PID: {proc.pid}, logs: {log_file})"
        except Exception as exc:
            ctx["error"] = type(exc).__name__
            log_error("failed to start server with nohup", ctx=ctx)
            # Don't fallback to blocking mode - raise error instead to prevent indefinite stalls
            raise RuntimeError(
                f"Failed to start server in background: {exc}\n"
                "The deploy command is designed to be non-blocking for AI agent use.\n"
                "If you need blocking mode, use the task app's serve command directly."
            ) from exc
    except Exception as exc:
        ctx["error"] = type(exc).__name__
        log_error("deploy_app_uvicorn failed", ctx=ctx)
        raise
