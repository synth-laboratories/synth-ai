from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
SYNTH_HOME_DIR = Path(os.environ.get("SYNTH_HOME", Path.home() / ".synth_ai")).expanduser()
SYNTH_USER_CONFIG_PATH = Path(
    os.environ.get("SYNTH_USER_CONFIG_PATH", SYNTH_HOME_DIR / "config.json")
).expanduser()
SYNTH_CONTAINER_CONFIG_PATH = Path(
    os.environ.get("SYNTH_CONTAINER_CONFIG_PATH", SYNTH_HOME_DIR / "container.json")
).expanduser()
SYNTH_BIN_DIR = Path(os.environ.get("SYNTH_BIN_DIR", SYNTH_HOME_DIR / "bin")).expanduser()


def is_file_type(path: Path | str, type: str) -> bool:
    return Path(path).suffix.lstrip(".").lower() == type.lower()


def validate_file_type(path: Path | str, type: str) -> None:
    if not is_file_type(path, type):
        raise ValueError(f"Invalid file type: expected .{type} got {Path(path).suffix}")


def is_hidden_path(path: Path | str, root: Path | str) -> bool:
    resolved_path = Path(path).expanduser()
    resolved_root = Path(root).expanduser()
    try:
        rel = resolved_path.resolve().relative_to(resolved_root.resolve())
    except Exception:
        rel = resolved_path
    return any(part.startswith(".") for part in rel.parts)


def get_bin_path(name: str) -> Path | None:
    which = shutil.which(name)
    if which:
        return Path(which)
    candidate = SYNTH_BIN_DIR / name
    return candidate if candidate.exists() else None


def get_home_config_file_paths(dir_name: str, file_extension: str = "json") -> list[Path]:
    base = SYNTH_HOME_DIR / dir_name
    if not base.exists():
        return []
    return sorted(base.glob(f"*.{file_extension}"))


def find_config_path(
    bin: Path,
    home_subdir: str,
    filename: str,
) -> Path | None:
    candidate = Path(bin).expanduser()
    if candidate.exists():
        return candidate
    candidate = Path.home() / home_subdir / filename
    return candidate if candidate.exists() else None


def configure_import_paths(app: Path, repo_root: Path | None = REPO_ROOT) -> None:
    paths = [str(Path(app).expanduser().resolve().parent)]
    if repo_root:
        paths.append(str(Path(repo_root).expanduser().resolve()))
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    for dir in reversed(paths):
        if dir and dir not in sys.path:
            sys.path.insert(0, dir)


@contextlib.contextmanager
def temporary_import_paths(app: Path, repo_root: Path | None = REPO_ROOT):
    """Temporarily configure PYTHONPATH/sys.path for loading a container from a file path."""
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
    file_path = Path(file).expanduser()
    dir_path = Path(dir).expanduser()
    if file_path.exists() and file_path.is_file():
        file_path.unlink()
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)


def print_paths_formatted(entries: list[tuple]) -> None:
    for i, entry in enumerate(entries, start=1):
        *item_parts, mtime = entry
        suffix = " ← most recent" if i == 1 else ""
        timestamp = f"modified {mtime}" if mtime else ""
        start = f"[{item_parts[0]}] {item_parts[1]}" if len(item_parts) == 2 else str(item_parts[0])
        print(f"{start}  |  {timestamp}{suffix}")
