import ast
import importlib.util as importlib
import py_compile
import subprocess
import sys
from ast import Module as AstModule
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from fastapi.routing import APIRoute
from starlette.types import ASGIApp


def run_ruff_check(path: Path, fix: bool = False) -> int:
    cmd = ["ruff", "check", "--extend-ignore", "I001,I002"]
    if fix:
        cmd.append("--fix")
    cmd.append(str(path))
    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        return 0


def extract_routes_from_app(app: object) -> list[str]:
    router = getattr(app, "router", None)
    if router is None:
        return []
    routes = getattr(router, "routes", None)
    if routes is None:
        return []
    return [getattr(route, "path", '') for route in routes]


def build_fastapi_route_index(app: ASGIApp) -> dict[str, list[APIRoute]]:
    router = getattr(app, "router", None)
    if router is None:
        raise ValueError("App does not expose a FastAPI router")
    entries: dict[str, list[APIRoute]] = {}
    for route in getattr(router, "routes", []):
        if isinstance(route, APIRoute):
            entries.setdefault(route.path, []).append(route)
    return entries


def get_asgi_app(module: ModuleType) -> ASGIApp:
    def _coerce_app(candidate: Any) -> ASGIApp | None:
        if candidate is None:
            return None
        if callable(candidate):
            return cast(ASGIApp, candidate)
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


def get_module(path: Path) -> AstModule:
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read file {path}: {exc}") from exc
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"{path} contains invalid Python syntax: {exc}") from exc


def load_module(
    path: Path,
    module_name: str | None = None
) -> ModuleType:
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


def validate_py_file_compiles(path: Path) -> None:
    path = path.resolve()
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        raise ValueError(f"Failed to compile {path}: {exc.msg}") from exc
    return None
