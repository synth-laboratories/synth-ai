import os
import threading

from synth_ai.cfgs import LocalDeployCfg
from synth_ai.utils import log_error, log_event
from synth_ai.utils.apps import get_asgi_app, load_file_to_module
from synth_ai.utils.paths import REPO_ROOT, configure_import_paths

import uvicorn
from uvicorn._types import ASGIApplication

_THREADS: dict[int, threading.Thread] = {}


def serve_app_uvicorn(
    app: ASGIApplication,
    host: str,
    port: int
) -> None:
    ctx = {"host": str(host), "port": int(port)}
    log_event("info", "starting uvicorn server", ctx=ctx)
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        log_event("info", "uvicorn server exited cleanly", ctx=ctx)
    except Exception as exc:
        log_error("uvicorn server failed", ctx={**ctx, "error": type(exc).__name__})
        raise RuntimeError(f"Failed to serve app with Uvicorn: {exc}") from exc


def deploy_app_uvicorn(cfg: LocalDeployCfg) -> str | None:
    ctx = {
        "task_app": str(cfg.task_app_path),
        "host": str(cfg.host),
        "port": int(cfg.port),
        "trace": bool(cfg.trace),
    }
    log_event("info", "deploy_app_uvicorn invoked", ctx=ctx)
    try:
        os.environ["ENVIRONMENT_API_KEY"] = cfg.env_api_key
        if cfg.trace:
            os.environ["TASKAPP_TRACING_ENABLED"] = "1"
        else:
            os.environ.pop("TASKAPP_TRACING_ENABLED", None)

        configure_import_paths(cfg.task_app_path, REPO_ROOT)
        module = load_file_to_module(
            cfg.task_app_path,
            f"_synth_local_task_app_{cfg.task_app_path.stem}"
        )
        log_event("info", "task app module loaded", ctx=ctx)
        app = get_asgi_app(module)

        msg = f"Serving task app at http://{'127.0.0.1' if cfg.host in {'0.0.0.0', '::'} else cfg.host}:{cfg.port}"
        if os.environ.get("CTX") == "mcp":
            thread = threading.Thread(
                target=serve_app_uvicorn,
                args=(app, cfg.host, cfg.port),
                name=f"synth-uvicorn-{cfg.port}",
                daemon=True
            )
            thread.start()
            _THREADS[cfg.port] = thread
            log_event("info", "uvicorn server running in background thread", ctx={**ctx, "mode": "mcp"})
            return f"[deploy_local] {msg}"
        print(msg)
        serve_app_uvicorn(app, cfg.host, cfg.port)
        log_event("info", "deploy_app_uvicorn completed", ctx=ctx)
        return None
    except Exception as exc:
        log_error("deploy_app_uvicorn failed", ctx={**ctx, "error": type(exc).__name__})
        raise
