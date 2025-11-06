from pathlib import Path
from types import ModuleType
from typing import Any, Set, cast

from synth_ai.utils.paths import load_file_to_module
from uvicorn._types import ASGIApplication


def extract_routes_from_app(app: object) -> list[str]:
    router = getattr(app, "router", None)
    if router is None:
        return []
    routes = getattr(router, "routes", None)
    if routes is None:
        return []
    return [getattr(route, "path", '') for route in routes]


def get_asgi_app(module: ModuleType) -> ASGIApplication:
    def _coerce_app(candidate: Any) -> ASGIApplication | None:
        if candidate is None:
            return None
        if callable(candidate):
            return cast(ASGIApplication, candidate)
        return None
    
    app = _coerce_app(getattr(module, "app", None))
    if app:
        return app
    
    asgi_factory_names = (
        "fastapi_app",
        "create_app",
        "build_app",
        "configure_app",
        "get_app",
        "app_factory",
    )
    for name in asgi_factory_names:
        factory = getattr(module, name, None)
        if callable(factory):
            produced = factory()
            coerced = _coerce_app(produced)
            if coerced is not None:
                app = coerced
                break

    if app is None:
        raise RuntimeError("Task app must expose an ASGI application via `app = FastAPI(...)` or a callable factory.")
    return app


def validate_task_app(path: Path) -> None:
    path = path.resolve()
    module = load_file_to_module(path)
    app = get_asgi_app(module)
    if app is None:
        raise ValueError(f"Failed to extract app attribute from module for {path}")
    routes = set(extract_routes_from_app(app))
    required_endpoints: Set[str] = {
        '/',
        "/health",
        "/info",
        "/task_info",
        "/rollout"
    }
    missing = required_endpoints - routes
    if missing:
        raise ValueError(f"{path} is missing required FastAPI endpoints: {', '.join(sorted(missing))}")
