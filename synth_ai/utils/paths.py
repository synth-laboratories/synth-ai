import shutil
from pathlib import Path


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
    file_extension: str = ".json"
) -> list[Path]:
    dir = Path.home() / dir_name
    if not dir.exists():
        return []
    return [path for path in dir.glob(f"*.{file_extension}") if path.is_file()]
