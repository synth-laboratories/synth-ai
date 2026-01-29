from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for utils.paths.") from exc


def _call_optional(name: str, *args, default=None):
    fn = getattr(synth_ai_py, name, None)
    if callable(fn):
        try:
            return fn(*args)
        except Exception:
            return default
    return default


_repo_root = _call_optional("repo_root")
REPO_ROOT = Path(_repo_root or Path.cwd())
SYNTH_HOME_DIR = Path(_call_optional("synth_home_dir", default=Path.home() / ".synth_ai"))
SYNTH_USER_CONFIG_PATH = Path(
    _call_optional("synth_user_config_path", default=SYNTH_HOME_DIR / "config.json")
)
SYNTH_LOCALAPI_CONFIG_PATH = Path(
    _call_optional("synth_localapi_config_path", default=SYNTH_HOME_DIR / "localapi.json")
)
SYNTH_BIN_DIR = Path(_call_optional("synth_bin_dir", default=SYNTH_HOME_DIR / "bin"))


def is_file_type(path: Path, type: str) -> bool:
    fn = getattr(synth_ai_py, "is_file_type", None)
    if callable(fn):
        return fn(str(path), type)
    return path.suffix.lstrip(".").lower() == type.lower()


def validate_file_type(path: Path, type: str) -> None:
    fn = getattr(synth_ai_py, "validate_file_type", None)
    if callable(fn):
        fn(str(path), type)
        return
    if not is_file_type(path, type):
        raise ValueError(f"Invalid file type: expected .{type} got {path.suffix}")


def is_hidden_path(path: Path, root: Path) -> bool:
    fn = getattr(synth_ai_py, "is_hidden_path", None)
    if callable(fn):
        return fn(str(path), str(root))
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        rel = path
    return any(part.startswith(".") for part in rel.parts)


def get_bin_path(name: str) -> Path | None:
    fn = getattr(synth_ai_py, "get_bin_path", None)
    if callable(fn):
        path = fn(name)
        return Path(path) if path else None
    which = shutil.which(name)
    if which:
        return Path(which)
    candidate = SYNTH_BIN_DIR / name
    return candidate if candidate.exists() else None


def get_home_config_file_paths(dir_name: str, file_extension: str = "json") -> list[Path]:
    fn = getattr(synth_ai_py, "get_home_config_file_paths", None)
    if callable(fn):
        return [Path(p) for p in fn(dir_name, file_extension)]
    base = SYNTH_HOME_DIR / dir_name
    if not base.exists():
        return []
    return sorted(base.glob(f"*.{file_extension}"))


def find_config_path(
    bin: Path,
    home_subdir: str,
    filename: str,
) -> Path | None:
    fn = getattr(synth_ai_py, "find_config_path", None)
    if callable(fn):
        path = fn(str(bin), home_subdir, filename)
        return Path(path) if path else None
    candidate = Path.home() / home_subdir / filename
    return candidate if candidate.exists() else None


def configure_import_paths(app: Path, repo_root: Path | None = REPO_ROOT) -> None:
    fn = getattr(synth_ai_py, "compute_import_paths", None)
    if callable(fn):
        paths = fn(str(app), str(repo_root) if repo_root else None)
    else:
        paths = [str(app.parent)]
        if repo_root:
            paths.append(str(repo_root))
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    for dir in reversed(paths):
        if dir and dir not in sys.path:
            sys.path.insert(0, dir)


@contextlib.contextmanager
def temporary_import_paths(app: Path, repo_root: Path | None = REPO_ROOT):
    """Temporarily configure PYTHONPATH/sys.path for loading a task app from a file path."""
    original_sys_path = sys.path.copy()
    original_pythonpath = os.environ.get("PYTHONPATH")
    configure_import_paths(app, repo_root)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath


def cleanup_paths(*, file: Path, dir: Path) -> None:
    fn = getattr(synth_ai_py, "cleanup_paths", None)
    if callable(fn):
        fn(str(file), str(dir))


def print_paths_formatted(entries: list[tuple]) -> None:
    for i, entry in enumerate(entries, start=1):
        *item_parts, mtime = entry
        suffix = " ← most recent" if i == 1 else ""
        timestamp = f"modified {mtime}" if mtime else ""
        start = f"[{item_parts[0]}] {item_parts[1]}" if len(item_parts) == 2 else str(item_parts[0])
        print(f"{start}  |  {timestamp}{suffix}")
