import importlib.util as importlib
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType


def is_py_file(path: Path) -> bool:
    path = path.resolve()
    if not path.is_file():
        return False
    return path.suffix == ".py"


def find_bin_path(name: str) -> Path | None:
    path = shutil.which(name)
    if not path:
        return None
    return Path(path)


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
    bin_path: Path,
    home_subdir: str,
    filename: str,
) -> Path | None:
    """
    Return a config file located in the user's home directory or alongside the binary.

    Args:
        bin_path: Resolved path to the executable.
        home_subdir: Directory under the user's home to inspect (e.g., ".codex").
        filename: Name of the config file to locate.
    """
    home_candidate = Path.home() / home_subdir / filename
    if home_candidate.exists():
        return home_candidate

    local_candidate = Path(bin_path).parent / home_subdir / filename
    if local_candidate.exists():
        return local_candidate

    return None


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


def configure_import_paths(
    app_path: Path,
    repo_root: Path | None = None
) -> None:
    app_dir = app_path.parent.resolve()

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
