from __future__ import annotations


class ServeCliError(RuntimeError):
    """Base exception for serve CLI failures."""


__all__ = ["ServeCliError"]
