from __future__ import annotations


class ModalServeCliError(RuntimeError):
    """Base exception for modal-serve CLI failures."""


__all__ = ["ModalServeCliError"]
