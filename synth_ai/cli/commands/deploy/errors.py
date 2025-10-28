from __future__ import annotations


class DeployCliError(RuntimeError):
    """Base exception for deploy CLI failures."""


__all__ = ["DeployCliError"]
