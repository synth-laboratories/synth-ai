import importlib.util as import_util
import os
import sys
from pathlib import Path
from typing import Any

from synth_ai.task_app_cfgs import LocalTaskAppConfig
from synth_ai.utils.env import resolve_env_var

REPO_ROOT = Path(__file__).resolve().parents[2]
START_DIV = f"{'-' * 30} Uvicorn start {'-' * 30}"
END_DIV = f"{'-' * 31} Uvicorn end {'-' * 31}"

_ASGI_FACTORY_NAMES = (
    "fastapi_app",
    "create_app",
    "build_app",
    "configure_app",
    "get_app",
    "app_factory",
)


def _coerce_asgi_app(candidate: Any) -> Any | None:
    if candidate is None:
        return None
    if callable(candidate):
        return candidate
    return None


def deploy_uvicorn_app(cfg: LocalTaskAppConfig) -> None:
    task_app_path = cfg.task_app_path.resolve()

    env_key = resolve_env_var("ENVIRONMENT_API_KEY")
    if not env_key:
        raise RuntimeError("ENVIRONMENT_API_KEY is required to serve locally.")

    if cfg.trace:
        os.environ["TASKAPP_TRACING_ENABLED"] = "1"
    else:
        os.environ.pop("TASKAPP_TRACING_ENABLED", None)

    task_app_dir = task_app_path.parent.resolve()
    candidates: list[Path] = [task_app_dir]
    if (task_app_dir / "__init__.py").exists():
        candidates.append(task_app_dir.parent.resolve())
    candidates.append(REPO_ROOT)

    unique: list[str] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str and candidate_str not in unique:
            unique.append(candidate_str)

    existing = os.environ.get("PYTHONPATH")
    if existing:
        for segment in existing.split(os.pathsep):
            if segment and segment not in unique:
                unique.append(segment)

    os.environ["PYTHONPATH"] = os.pathsep.join(unique)
    for entry in reversed(unique):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)

    module_name = f"_synth_local_task_app_{task_app_path.stem}"
    spec = import_util.spec_from_file_location(module_name, str(task_app_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load task app at {task_app_path}")
    module = import_util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[call-arg]
    except Exception as exc:
        raise RuntimeError(f"Failed to import task app: {exc}") from exc

    app = _coerce_asgi_app(getattr(module, "app", None))
    if app is None:
        for name in _ASGI_FACTORY_NAMES:
            factory = getattr(module, name, None)
            if callable(factory):
                produced = factory()
                coerced = _coerce_asgi_app(produced)
                if coerced is not None:
                    app = coerced
                    break
    if app is None:
        raise RuntimeError("Task app must expose an ASGI application via `app = FastAPI(...)` or a callable factory.")

    host = cfg.host
    port = cfg.port
    preview_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"[uvicorn] Serving task app at http://{preview_host}:{port}")


# Deploy
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to serve task apps locally. Install it with `pip install uvicorn`."
        ) from exc

    try:
        print(START_DIV)
        uvicorn.run(app, host=host, port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[uvicorn] Stopped by user.")
    except Exception as exc:
        raise RuntimeError(f"uvicorn runtime failed: {exc}") from exc
    finally:
        print(END_DIV)
