from __future__ import annotations

import os
from pathlib import Path
from typing import Any
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
    if lowered in {"dev", "development", "staging", "railway"}:
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
    explicit = (
        (
            os.getenv("ENVIRONMENT")
            or os.getenv("APP_ENVIRONMENT")
            or os.getenv("RAILWAY_ENVIRONMENT")
            or os.getenv("RAILWAY_ENVIRONMENT_NAME")
            or os.getenv("ENV")
            or ""
        )
        .strip()
        .lower()
    )
    if explicit:
        return explicit
    if os.getenv("PROD_SYNTH_BACKEND_URL") or os.getenv("PROD_BACKEND_URL"):
        return "prod"
    if os.getenv("DEV_SYNTH_BACKEND_URL") or os.getenv("DEV_BACKEND_URL"):
        return "dev"
    return "dev"


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


def normalize_inference_base(url: str) -> str:
    parsed = urlparse(str(url).strip())
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if path.endswith(suffix):
            path = _strip_terminal_segment(path, suffix)
            break
    normalized = parsed._replace(path=path.rstrip("/"), fragment="")
    return urlunparse(normalized)


def normalize_base_url(url: str) -> str:
    return normalize_backend_base(url)


def resolve_synth_backend_url(override: str | None = None) -> str:
    if override and override.strip():
        coerced = _coerce_backend_override(override)
        if coerced:
            return normalize_backend_base(coerced)
        if _looks_like_url(override):
            return normalize_backend_base(override)
    return BACKEND_URL_BASE


def resolve_synth_interceptor_base_url(override: str | None = None) -> str:
    return join_url(resolve_synth_backend_url(override), "/api/interceptor/v1").rstrip("/")


def local_backend_url(host: str = "localhost", port: int = 8000) -> str:
    return f"http://{host}:{port}"


def backend_health_url(base_url: str) -> str:
    return join_url(base_url, "/health")


def backend_me_url(base_url: str) -> str:
    return join_url(base_url, "/api/v1/me")


def backend_demo_keys_url(base_url: str) -> str:
    return join_url(base_url, "/api/demo/keys")


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


def _host_matches_pattern(host: str, pattern: str) -> bool:
    if not pattern:
        return False
    if pattern.startswith("*."):
        suffix = pattern[1:]
        return host.endswith(suffix) and len(host) > len(suffix)
    if pattern.startswith("."):
        return host.endswith(pattern)
    return host == pattern


def is_cloudflare_tunnel_url(url: str) -> bool:
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return hostname.endswith(".trycloudflare.com") or hostname.endswith(".cfargotunnel.com")


def is_free_ngrok_url(url: str) -> bool:
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return hostname.endswith(".ngrok-free.app") or hostname.endswith(".ngrok-free.dev")


def is_local_http_container_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "http":
        return False
    return is_local_hostname(parsed.hostname)


def is_synth_managed_ngrok_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "https":
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname or is_cloudflare_tunnel_url(url) or is_free_ngrok_url(url):
        return False
    patterns = ["usesynth.ai", "*.usesynth.ai"]
    extra = (os.getenv("SYNTH_MANAGED_TUNNEL_TRUSTED_HOSTS") or "").strip().lower()
    if extra:
        for raw in extra.split(","):
            value = raw.strip()
            if value and value not in patterns:
                patterns.append(value)
    return any(_host_matches_pattern(hostname, pattern) for pattern in patterns)


def is_synthtunnel_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if not parsed.path.startswith("/s/"):
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    patterns = [
        "st.usesynth.ai",
        "*.st.usesynth.ai",
        "api-dev.usesynth.ai",
        "api.usesynth.ai",
        "localhost",
        "127.0.0.1",
        "::1",
    ]
    extra = (os.getenv("SYNTH_TUNNEL_TRUSTED_HOSTS") or "").strip().lower()
    if extra:
        for raw in extra.split(","):
            value = raw.strip()
            if value and value not in patterns:
                patterns.append(value)
    return any(_host_matches_pattern(hostname, pattern) for pattern in patterns)


def extract_prompt_learning_container_url(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    prompt_learning = payload.get("prompt_learning")
    if isinstance(prompt_learning, dict):
        value = prompt_learning.get("container_url")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = payload.get("container_url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def infer_prompt_learning_container_url(
    *,
    overrides: dict[str, Any] | None = None,
    config_dict: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> str | None:
    if isinstance(overrides, dict):
        value = overrides.get("container_url")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(config_dict, dict):
        url = extract_prompt_learning_container_url(config_dict)
        if url:
            return url
    if config_path:
        try:
            import tomllib

            with Path(config_path).open("rb") as fh:
                payload = tomllib.load(fh)
            url = extract_prompt_learning_container_url(payload)
            if url:
                return url
        except Exception:
            pass
    env_url = (os.environ.get("CONTAINER_URL") or "").strip()
    return env_url or None


BACKEND_URL_BASE = normalize_backend_base(_resolve_backend_url())
BACKEND_URL_API = join_url(BACKEND_URL_BASE, "/api")
BACKEND_URL_SYNTH_RESEARCH_BASE = join_url(BACKEND_URL_BASE, "/api/synth-research")
BACKEND_URL_SYNTH_RESEARCH_OPENAI = join_url(BACKEND_URL_SYNTH_RESEARCH_BASE, "/v1")
BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC = BACKEND_URL_SYNTH_RESEARCH_BASE
FRONTEND_URL_BASE = _env_or_default("SYNTH_FRONTEND_URL", "https://usesynth.ai")


__all__ = [
    "BACKEND_URL_API",
    "BACKEND_URL_BASE",
    "BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC",
    "BACKEND_URL_SYNTH_RESEARCH_BASE",
    "BACKEND_URL_SYNTH_RESEARCH_OPENAI",
    "FRONTEND_URL_BASE",
    "LOCAL_HTTP_HOSTS",
    "backend_demo_keys_url",
    "backend_health_url",
    "backend_me_url",
    "extract_prompt_learning_container_url",
    "infer_prompt_learning_container_url",
    "is_cloudflare_tunnel_url",
    "is_free_ngrok_url",
    "is_local_backend_base_url",
    "is_local_hostname",
    "is_local_http_container_url",
    "is_synth_managed_ngrok_url",
    "is_synthtunnel_url",
    "join_url",
    "local_backend_url",
    "normalize_backend_base",
    "normalize_base_url",
    "normalize_inference_base",
    "resolve_synth_backend_url",
    "resolve_synth_interceptor_base_url",
]
