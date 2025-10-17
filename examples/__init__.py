"""Top-level package for Synth AI example environments and utilities."""

from importlib import resources as _resources

__all__ = ["path_for"]


def path_for(package: str, resource: str) -> str:
    """Return absolute path for a packaged resource inside ``examples``.

    This helper mirrors the one under ``synth_ai`` so hosted apps can access
    bundled assets without needing to install the repo in editable mode.
    """

    with _resources.as_file(_resources.files(f"examples.{package}") / resource) as path:
        return str(path)
