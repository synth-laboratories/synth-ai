import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def is_file_type(path: Path, type: str) -> bool:
    if not type.startswith('.'):
        type = '.' + type
    return path.is_file() and path.suffix == type


def validate_file_type(path: Path, type: str) -> None:
    if not is_file_type(path, type):
        raise ValueError(f"{path} is not a {type} file")
    return None


def is_hidden_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part.startswith('.') for part in relative.parts)


def get_bin_path(name: str) -> Path | None:
    path = shutil.which(name)
    return Path(path) if path else None


def get_env_file_paths(base_dir: str | Path = '.') -> list[Path]:
    base = Path(base_dir).resolve()
    return [path for path in base.rglob(".env*") if path.is_file()]


def get_home_config_file_paths(
    dir_name: str,
    file_extension: str = "json"
) -> list[Path]:
    dir = Path.home() / dir_name
    if not dir.exists():
        return []
    return [path for path in dir.glob(f"*.{file_extension}") if path.is_file()]


def find_config_path(
    bin: Path,
    home_subdir: str,
    filename: str,
) -> Path | None:
    home_candidate = Path.home() / home_subdir / filename
    if home_candidate.exists():
        return home_candidate
    local_candidate = Path(bin).parent / home_subdir / filename
    if local_candidate.exists():
        return local_candidate
    return None


def configure_import_paths(
    app: Path,
    repo_root: Path | None = REPO_ROOT
) -> None:
    app_dir = app.parent.resolve()

    initial_dirs: list[Path] = [app_dir]
    if (app_dir / "__init__.py").exists():
        initial_dirs.append(app_dir.parent.resolve())
    if repo_root:
        initial_dirs.append(repo_root)

    unique_dirs: list[str] = []
    for dir in initial_dirs:
        dir_str = str(dir)
        if dir_str and dir_str not in unique_dirs:
            unique_dirs.append(dir_str)

    existing_pythonpath_dirs = os.environ.get("PYTHONPATH")
    if existing_pythonpath_dirs:
        for segment in existing_pythonpath_dirs.split(os.pathsep):
            if segment and segment not in unique_dirs:
                unique_dirs.append(segment)

    os.environ["PYTHONPATH"] = os.pathsep.join(unique_dirs)

    for dir in reversed(unique_dirs):
        if dir and dir not in sys.path:
            sys.path.insert(0, dir)


def cleanup_paths(*, file: Path, dir: Path) -> None:
    if not file.is_relative_to(dir):
        raise ValueError(f"{file} is not inside {dir}")
    file.unlink(missing_ok=True)
    shutil.rmtree(dir, ignore_errors=True)


def print_paths_formatted(entries: list[tuple]) -> None:
    for i, entry in enumerate(entries, start=1):
        *item_parts, mtime = entry
        suffix = " ← most recent" if i == 1 else ""
        timestamp = f"modified {mtime}" if mtime else ""
        start = f"[{item_parts[0]}] {item_parts[1]}" if len(item_parts) == 2 else str(item_parts[0])
        print(f"{start}  |  {timestamp}{suffix}")
