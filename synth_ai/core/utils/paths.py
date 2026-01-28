from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for utils.paths.") from exc

REPO_ROOT = Path(synth_ai_py.repo_root() or Path.cwd())
SYNTH_HOME_DIR = Path(synth_ai_py.synth_home_dir())
SYNTH_USER_CONFIG_PATH = Path(synth_ai_py.synth_user_config_path())
SYNTH_LOCALAPI_CONFIG_PATH = Path(synth_ai_py.synth_localapi_config_path())
SYNTH_BIN_DIR = Path(synth_ai_py.synth_bin_dir())


def is_file_type(path: Path, type: str) -> bool:
    return synth_ai_py.is_file_type(str(path), type)


def validate_file_type(path: Path, type: str) -> None:
    synth_ai_py.validate_file_type(str(path), type)


def is_hidden_path(path: Path, root: Path) -> bool:
    return synth_ai_py.is_hidden_path(str(path), str(root))


def get_bin_path(name: str) -> Path | None:
    path = synth_ai_py.get_bin_path(name)
    return Path(path) if path else None


def get_home_config_file_paths(dir_name: str, file_extension: str = "json") -> list[Path]:
    return [Path(p) for p in synth_ai_py.get_home_config_file_paths(dir_name, file_extension)]


def find_config_path(
    bin: Path,
    home_subdir: str,
    filename: str,
) -> Path | None:
    path = synth_ai_py.find_config_path(str(bin), home_subdir, filename)
    return Path(path) if path else None


def configure_import_paths(app: Path, repo_root: Path | None = REPO_ROOT) -> None:
    paths = synth_ai_py.compute_import_paths(str(app), str(repo_root) if repo_root else None)
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
    synth_ai_py.cleanup_paths(str(file), str(dir))


def print_paths_formatted(entries: list[tuple]) -> None:
    for i, entry in enumerate(entries, start=1):
        *item_parts, mtime = entry
        suffix = " ← most recent" if i == 1 else ""
        timestamp = f"modified {mtime}" if mtime else ""
        start = f"[{item_parts[0]}] {item_parts[1]}" if len(item_parts) == 2 else str(item_parts[0])
        print(f"{start}  |  {timestamp}{suffix}")
