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
