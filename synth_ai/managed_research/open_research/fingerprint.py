"""Anonymous-submitter fingerprint persistence.

Open Research's 1h open-ended-discovery queue admits unsigned-in
submissions. The backend identifies these callers from an opaque
``X-OR-Fingerprint`` header (and mirrors it into
``submitter.fingerprint``). The MCP caller can supply one explicitly;
otherwise we generate and persist a stable per-machine value so the
backend's idempotency key (``submitter_fingerprint, project_slug,
prompt_hash``) deduplicates retries from the same workstation.

Storage path:

- ``$MANAGED_RESEARCH_OR_FINGERPRINT_PATH`` if set, else
- ``$XDG_CACHE_HOME/synth_ai.managed_research/or_fingerprint`` if XDG is set, else
- ``~/.cache/synth_ai.managed_research/or_fingerprint``.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path


def _resolve_default_path() -> Path:
    override = os.environ.get("MANAGED_RESEARCH_OR_FINGERPRINT_PATH")
    if override and override.strip():
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg and xdg.strip() else Path.home() / ".cache"
    return base / "synth_ai.managed_research" / "or_fingerprint"


DEFAULT_FINGERPRINT_PATH: Path = _resolve_default_path()


def load_or_create_fingerprint(path: Path | None = None) -> str:
    """Return the persisted fingerprint or mint and persist a fresh one.

    One correct path. We never silently fall back to a session-scoped or
    in-memory value when persistence fails — the caller sees the real
    OSError so they can fix permissions or supply the value explicitly.
    """
    target = path if path is not None else _resolve_default_path()
    if target.exists():
        existing = target.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    minted = uuid.uuid4().hex
    target.write_text(minted + "\n", encoding="utf-8")
    # Filesystems without chmod (FAT/network mounts) still get the
    # value persisted; tightening permissions is best-effort.
    with contextlib.suppress(OSError):
        target.chmod(0o600)
    return minted


__all__ = ["DEFAULT_FINGERPRINT_PATH", "load_or_create_fingerprint"]
