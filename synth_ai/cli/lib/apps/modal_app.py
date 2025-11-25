import ast
from datetime import datetime
from pathlib import Path

from synth_ai.cli.lib.prompts import ctx_print
from synth_ai.core.apps.common import get_module, validate_py_file_compiles
from synth_ai.core.paths import is_hidden_path, validate_file_type


def get_app_name(call: ast.Call) -> str | None:
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


def validate_app_declaration(path: Path) -> None:
    ast_module = get_module(path)
    app_aliases: set[str] = set()
    modal_aliases: set[str] = set()
    for node in ast.walk(ast_module):
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
                if get_app_name(node):
                    return None
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "App"
                and isinstance(func.value, ast.Name)
                and func.value.id in modal_aliases
                and get_app_name(node)
            ):
                return None
    raise ValueError(f"{path} must declare `app = modal.App(...)` (or import `App` directly) with a literal name.")


def validate_modal_app(
    path: Path,
    discovery: bool = False
) -> Path:
    def print_pass():
        ctx_print("Check passed", not discovery)
    
    ctx_print("\nChecking if .py file", not discovery)
    validate_file_type(path, ".py")
    print_pass()
    
    ctx_print("\nChecking if app = modal.App(...) declaration exists", not discovery)
    validate_app_declaration(path)
    print_pass()

    ctx_print("\nChecking if compiles", not discovery)
    validate_py_file_compiles(path)
    print_pass()
    print('\n')

    return path


def find_modal_apps_in_cwd() -> list[tuple[Path, str]]:
    cwd = Path.cwd().resolve()
    entries: list[tuple[Path, str, float]] = []
    for path in cwd.rglob("*.py"):
        if is_hidden_path(path, cwd):
            continue
        if not path.is_file():
            continue
        try:
            validate_modal_app(path, True)
        except Exception:
            continue
        try:
            rel_path = path.relative_to(cwd)
        except ValueError:
            rel_path = path
        try:
            mtime = path.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime).isoformat(sep=" ", timespec="seconds")
        except OSError:
            mtime = 0.0
            mtime_str = ""
        entries.append((rel_path, mtime_str, mtime))
    entries.sort(key=lambda item: item[2], reverse=True)
    return [(rel_path, mtime_str) for rel_path, mtime_str, _ in entries]
