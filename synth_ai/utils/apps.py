import ast
import importlib.util as importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Set, cast

from synth_ai.utils.paths import is_py_file
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


def load_file_to_module(
    path: Path,
    module_name: str | None = None
) -> ModuleType:
    if not is_py_file(path):
        raise ValueError(f"{path} is not a .py file")
    name = module_name or path.stem
    spec = importlib.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.module_from_spec(spec)
    if module_name:
        sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        if module_name:
            sys.modules.pop(module_name, None)
        raise ImportError(f"Failed to import module: {exc}") from exc
    return module


def validate_task_app(path: Path | None) -> Path:
    assert path is not None
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
    return path


def validate_modal_app(path: Path | None) -> Path:
    assert path is not None
    if not is_py_file(path):
        raise ValueError(f"{path} must be a .py file containing a Modal app definition.")
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read modal app file {path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"{path} contains invalid Python syntax: {exc}") from exc

    app_aliases: set[str] = set()
    modal_aliases: set[str] = set()

    def literal_name(call: ast.Call) -> str | None:
        for kw in call.keywords:
            if (
                kw.arg in {"name", "app_name"}
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                return kw.value.value
        if call.args:
            first = call.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return first.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "modal":
            for alias in node.names:
                if alias.name == "App":
                    app_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "modal":
                    modal_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in app_aliases:
                if literal_name(node):
                    return path
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "App"
                and isinstance(func.value, ast.Name)
                and func.value.id in modal_aliases
                and literal_name(node)
            ):
                return path
    raise ValueError(f"{path} must declare `app = modal.App(...)` (or import `App` directly) with a literal name.")
