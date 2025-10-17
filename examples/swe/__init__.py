"""SWE task app examples package."""

from importlib import resources as _resources

__all__ = ["path_for"]


def path_for(package: str, resource: str) -> str:
    """Return path for packaged SWE example resources."""

    with _resources.as_file(_resources.files(f"examples.swe.{package}") / resource) as path:
        return str(path)
