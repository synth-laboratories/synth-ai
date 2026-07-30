from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

LOCAL_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal", "::1"})


def _env_or_default(key: str, default: str) -> str:
    value = (os.getenv(key) or "").strip()
    return value or default


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _looks_like_url(value: str) -> bool:
    raw = value.strip().lower()
    return raw.startswith("http://") or raw.startswith("https://")


def _strip_terminal_segment(path: str, segment: str) -> str:
    trimmed = path.rstrip("/")
    if trimmed.endswith(segment):
        return trimmed[: -len(segment)].rstrip("/")
    return trimmed


def _coerce_backend_override(value: str) -> str | None:
    raw = _strip_quotes(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"local", "localhost"}:
        return (os.getenv("LOCAL_BACKEND_URL") or "http://localhost:8000").strip()
    if lowered in {"dev", "development", "staging"}:
        return (
            os.getenv("DEV_SYNTH_BACKEND_URL")
            or os.getenv("DEV_BACKEND_URL")
            or "https://api-dev.usesynth.ai"
        ).strip()
    if lowered in {"prod", "production", "main"}:
        return (
            os.getenv("PROD_SYNTH_BACKEND_URL")
            or os.getenv("PROD_BACKEND_URL")
            or "https://api.usesynth.ai"
        ).strip()
    if _looks_like_url(raw):
        return raw
    return None


def _resolve_backend_url_override() -> str | None:
    override = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE") or "").strip()
    if not override:
        return None
    return _coerce_backend_override(override)


def _current_env() -> str:
    # Deliberately generic. Host-provider environment variables used to be read
    # here, which put Synth's own deployment platform into a package customers
    # install -- names they would never set and cannot act on. Services running
    # on such a platform set ENVIRONMENT explicitly; the backend does its own
    # provider detection in config.py.
    explicit = (
        (os.getenv("ENVIRONMENT") or os.getenv("APP_ENVIRONMENT") or os.getenv("ENV") or "")
        .strip()
        .lower()
    )
    if explicit:
        return explicit
    if os.getenv("PROD_SYNTH_BACKEND_URL") or os.getenv("PROD_BACKEND_URL"):
        return "prod"
    if os.getenv("DEV_SYNTH_BACKEND_URL") or os.getenv("DEV_BACKEND_URL"):
        return "dev"
    # Unconfigured means production. `SynthClient()` with no base_url is the
    # pip-install path: someone set SYNTH_API_KEY and called it, and resolving
    # that to localhost fails against a machine running no backend. Local
    # development is the case that says so -- via base_url, ENVIRONMENT,
    # SYNTH_BACKEND_URL_OVERRIDE, or the DEV_*/LOCAL_* variables.
    return "prod"


def _is_prod_environment(value: str) -> bool:
    return value in {"prod", "production", "main"}


def _resolve_backend_url() -> str:
    override = _resolve_backend_url_override()
    if override:
        return override
    if _is_prod_environment(_current_env()):
        return (
            os.getenv("PROD_SYNTH_BACKEND_URL")
            or os.getenv("SYNTH_BACKEND_URL")
            or os.getenv("SYNTH_API_URL")
            or os.getenv("PROD_BACKEND_URL")
            or os.getenv("BACKEND_URL")
            or "https://api.usesynth.ai"
        ).strip()
    return (
        os.getenv("DEV_SYNTH_BACKEND_URL")
        or os.getenv("SYNTH_BACKEND_URL")
        or os.getenv("SYNTH_API_URL")
        or os.getenv("DEV_BACKEND_URL")
        or os.getenv("BACKEND_URL")
        or "http://localhost:8000"
    ).strip()


def join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not path:
        return base
    if path.startswith("/"):
        return f"{base}{path}"
    return f"{base}/{path}"


def normalize_backend_base(url: str) -> str:
    parsed = urlparse(str(url).strip())
    path = _strip_terminal_segment(parsed.path, "/v1")
    path = _strip_terminal_segment(path, "/api")
    normalized = parsed._replace(path=path.rstrip("/"), query="", fragment="")
    return urlunparse(normalized)


def resolve_synth_backend_url(override: str | None = None) -> str:
    if override and override.strip():
        coerced = _coerce_backend_override(override)
        if coerced:
            return normalize_backend_base(coerced)
        if _looks_like_url(override):
            return normalize_backend_base(override)
    return BACKEND_URL_BASE


def is_local_hostname(host: str | None) -> bool:
    return str(host or "").strip().lower() in LOCAL_HTTP_HOSTS


def is_local_backend_base_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(str(url).strip())
    except Exception:
        return False
    return is_local_hostname(parsed.hostname)


BACKEND_URL_BASE = normalize_backend_base(_resolve_backend_url())
BACKEND_URL_SYNTH_RESEARCH_BASE = join_url(BACKEND_URL_BASE, "/api/synth-research")

__all__ = [
    "BACKEND_URL_BASE",
    "BACKEND_URL_SYNTH_RESEARCH_BASE",
    "LOCAL_HTTP_HOSTS",
    "is_local_backend_base_url",
    "is_local_hostname",
    "join_url",
    "normalize_backend_base",
    "resolve_synth_backend_url",
]
