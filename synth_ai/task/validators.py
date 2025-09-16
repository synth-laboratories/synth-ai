from __future__ import annotations

from urllib.parse import urlparse


def validate_task_app_url(url: str, *, name: str = "TASK_APP_BASE_URL") -> None:
    """Validate a Task App base URL (scheme + host present)."""

    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise ValueError(f"Invalid {name}: malformed: {url}")

