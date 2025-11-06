from typing import Set
from pathlib import Path
from synth_ai.utils import load_file_to_module


REQUIRED_ENDPOINTS: Set[str] = {
    '/',
    "/health",
    "/info",
    "/task_info",
    "/rollout"
}


def _extract_routes_from_app(app: object) -> list[str]:
    router = getattr(app, "router", None)
    if router is None:
        return []
    routes = getattr(router, "routes", None)
    if routes is None:
        return []
    return [getattr(route, "path", '') for route in routes]


def validate_task_app(path: Path) -> None:
    path = path.resolve()
    module = load_file_to_module(path)
    app = getattr(module, "app", None)
    if app is None:
        raise ValueError(f"Failed to extract app attribute from module for {path}")
    routes = set(_extract_routes_from_app(app))
    missing = REQUIRED_ENDPOINTS - routes
    if missing:
        raise ValueError(f"{path} is missing required FastAPI endpoints: {', '.join(sorted(missing))}")
