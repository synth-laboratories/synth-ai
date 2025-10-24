from __future__ import annotations

__all__ = ["key_preview"]


def key_preview(value: str, label: str) -> str:
    """Return a short descriptor for a secret without leaking the full value."""
    try:
        text = value or ""
        length = len(text)
        prefix = text[:6] if length >= 6 else text
        suffix = text[-5:] if length >= 5 else text
        return f"{label} len={length} prefix={prefix} last5={suffix}"
    except Exception:
        return f"{label} len=0"
