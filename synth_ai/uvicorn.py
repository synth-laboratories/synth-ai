import os
import threading

from synth_ai.cfgs import LocalDeployCfg
from synth_ai.utils.apps import get_asgi_app, load_file_to_module
from synth_ai.utils.paths import REPO_ROOT, configure_import_paths
from uvicorn._types import ASGIApplication

import uvicorn

_THREADS: dict[int, threading.Thread] = {}


def serve_app_uvicorn(
    app: ASGIApplication,
    host: str,
    port: int
) -> None:
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to serve app with Uvicorn: {exc}") from exc


def deploy_app_uvicorn(cfg: LocalDeployCfg) -> str | None:
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
        return f"[deploy_local] {msg}"
    print(msg)
    serve_app_uvicorn(app, cfg.host, cfg.port)
    return None
