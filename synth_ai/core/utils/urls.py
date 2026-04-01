from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse, urlunparse

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for URL utilities.") from exc


LOCAL_HTTP_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal", "::1"}
)


def _env_or_default(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value and value.strip() else default


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _looks_like_url(value: str) -> bool:
    v = value.strip()
    return v.startswith("http://") or v.startswith("https://")


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
    if (
        os.getenv("PROD_SYNTH_BACKEND_URL")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("PROD_RUST_BACKEND_URL")
        or os.getenv("RUST_BACKEND_URL")
    ):
        return "prod"
    if (
        os.getenv("DEV_SYNTH_BACKEND_URL")
        or os.getenv("DEV_BACKEND_URL")
        or os.getenv("DEV_RUST_BACKEND_URL")
    ):
        return "dev"
    return "dev"


def _is_prod_environment(value: str) -> bool:
    return value in {"prod", "production", "main"}


def _resolve_backend_url() -> str:
    override = _resolve_backend_url_override()
    if override:
        return override
    env = _current_env()
    if _is_prod_environment(env):
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


def _resolve_rust_backend_url() -> str:
    override = (os.getenv("SYNTH_RUST_BACKEND_URL_OVERRIDE") or "").strip()
    if override:
        raw = _strip_quotes(override).strip()
        lowered = raw.lower()
        if lowered in {"local", "localhost"}:
            return (
                os.getenv("LOCAL_RUST_BACKEND_URL")
                or os.getenv("RUST_BACKEND_URL")
                or "http://localhost:8080"
            ).strip()
        if lowered in {"dev", "development", "staging", "railway"}:
            return (
                os.getenv("DEV_RUST_BACKEND_URL")
                or os.getenv("RUST_BACKEND_URL")
                or "https://infra-api-dev.usesynth.ai"
            ).strip()
        if lowered in {"prod", "production", "main"}:
            return (
                os.getenv("PROD_RUST_BACKEND_URL")
                or os.getenv("RUST_BACKEND_URL")
                or "https://infra-api.usesynth.ai"
            ).strip()
        if _looks_like_url(raw):
            return raw

    env = _current_env()
    if _is_prod_environment(env):
        return (
            os.getenv("PROD_RUST_BACKEND_URL")
            or os.getenv("SYNTH_RUST_BACKEND_URL")
            or os.getenv("RUST_BACKEND_URL")
            or "https://infra-api.usesynth.ai"
        ).strip()
    return (
        os.getenv("DEV_RUST_BACKEND_URL")
        or os.getenv("SYNTH_RUST_BACKEND_URL")
        or os.getenv("RUST_BACKEND_URL")
        or "http://localhost:8080"
    ).strip()


def _maybe_call(name: str, default: str) -> str:
    fn = getattr(synth_ai_py, name, None)
    if callable(fn):
        return fn()
    return default


def _strip_terminal_segment(path: str, segment: str) -> str:
    trimmed = path.rstrip("/")
    if trimmed.endswith(segment):
        return trimmed[: -len(segment)].rstrip("/")
    return trimmed


def _normalize_backend_base_py(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    path = _strip_terminal_segment(path, "/v1")
    path = _strip_terminal_segment(path, "/api")
    normalized = parsed._replace(path=path.rstrip("/"), query="", fragment="")
    return urlunparse(normalized)


def _normalize_inference_base_py(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if path.endswith(suffix):
            path = _strip_terminal_segment(path, suffix)
            break
    normalized = parsed._replace(path=path.rstrip("/"), fragment="")
    return urlunparse(normalized)


def join_url(base_url: str, path: str) -> str:
    fn = getattr(synth_ai_py, "join_url", None)
    if callable(fn):
        return fn(base_url, path)
    base = base_url.rstrip("/")
    if not path:
        return base
    if path.startswith("/"):
        return f"{base}{path}"
    return f"{base}/{path}"


def _join_url_default(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not path:
        return base
    if path.startswith("/"):
        return f"{base}{path}"
    return f"{base}/{path}"


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


BACKEND_URL_BASE = _maybe_call(
    "backend_url_base",
    _resolve_backend_url(),
)
BACKEND_URL_API = _maybe_call("backend_url_api", _join_url_default(BACKEND_URL_BASE, "/api"))
BACKEND_URL_SYNTH_RESEARCH_BASE = _maybe_call(
    "backend_url_synth_research_base",
    _join_url_default(BACKEND_URL_BASE, "/api/synth-research"),
)
BACKEND_URL_SYNTH_RESEARCH_OPENAI = _maybe_call(
    "backend_url_synth_research_openai",
    _join_url_default(BACKEND_URL_SYNTH_RESEARCH_BASE, "/v1"),
)
BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC = _maybe_call(
    "backend_url_synth_research_anthropic", BACKEND_URL_SYNTH_RESEARCH_BASE
)
FRONTEND_URL_BASE = _maybe_call(
    "frontend_url_base", _env_or_default("SYNTH_FRONTEND_URL", "https://usesynth.ai")
)
RUST_BACKEND_URL_BASE = _maybe_call(
    "rust_backend_url_base",
    _resolve_rust_backend_url(),
)


def resolve_synth_backend_url(override: str | None = None) -> str:
    """Resolve backend base URL with environment-aware defaults."""
    if override and override.strip():
        coerced = _coerce_backend_override(override)
        if coerced:
            return normalize_backend_base(coerced)
        # If it's not a known token and not a URL, ignore it to avoid returning nonsense like "dev".
        if _looks_like_url(override):
            return normalize_backend_base(override)
    return BACKEND_URL_BASE


def resolve_synth_interceptor_base_url(override: str | None = None) -> str:
    """Resolve Synth interceptor base URL from a backend base URL."""
    return join_url(resolve_synth_backend_url(override), "/api/interceptor/v1").rstrip("/")


def normalize_base_url(url: str) -> str:
    return normalize_backend_base(url)


def normalize_backend_base(url: str) -> str:
    """Normalize backend base URL via the Rust core when available."""
    fn = getattr(synth_ai_py, "normalize_backend_base", None)
    if callable(fn):
        return fn(url)
    return _normalize_backend_base_py(url)


def normalize_inference_base(url: str) -> str:
    """Normalize inference base URL via the Rust core when available."""
    fn = getattr(synth_ai_py, "normalize_inference_base", None)
    if callable(fn):
        return fn(url)
    return _normalize_inference_base_py(url)


def local_backend_url(host: str = "localhost", port: int = 8000) -> str:
    fn = getattr(synth_ai_py, "local_backend_url", None)
    if callable(fn):
        return fn(host, port)
    return f"http://{host}:{port}"


def backend_health_url(base_url: str) -> str:
    fn = getattr(synth_ai_py, "backend_health_url", None)
    if callable(fn):
        return fn(base_url)
    return join_url(base_url, "/health")


def backend_me_url(base_url: str) -> str:
    fn = getattr(synth_ai_py, "backend_me_url", None)
    if callable(fn):
        return fn(base_url)
    return join_url(base_url, "/api/v1/me")


def backend_demo_keys_url(base_url: str) -> str:
    fn = getattr(synth_ai_py, "backend_demo_keys_url", None)
    if callable(fn):
        return fn(base_url)
    return join_url(base_url, "/api/demo/keys")


def is_synthtunnel_url(url: str) -> bool:
    """Return True if the URL targets the SynthTunnel gateway."""
    fn = getattr(synth_ai_py, "container_is_synthtunnel_url", None)
    if callable(fn):
        return bool(fn(url))
    return _is_synthtunnel_url_py(url)


def extract_prompt_learning_container_url(payload: dict[str, Any]) -> str | None:
    """Extract container URL from canonical prompt-learning config shapes."""
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
    """Infer container URL from overrides, config payloads, config file, then env."""
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
            payload = synth_ai_py.load_toml(str(config_path))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            url = extract_prompt_learning_container_url(payload)
            if url:
                return url

    env_url = (os.environ.get("CONTAINER_URL") or "").strip()
    return env_url or None


def is_cloudflare_tunnel_url(url: str) -> bool:
    """Return True if URL hostname indicates a Cloudflare tunnel."""
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return hostname.endswith(".trycloudflare.com") or hostname.endswith(".cfargotunnel.com")


def is_free_ngrok_url(url: str) -> bool:
    """Return True if URL hostname indicates a free/public ngrok endpoint."""
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    free_suffixes = (
        ".ngrok-free.app",
        ".ngrok-free.dev",
    )
    return any(hostname.endswith(suffix) for suffix in free_suffixes)


def _host_matches_pattern(host: str, pattern: str) -> bool:
    if not pattern:
        return False
    if pattern.startswith("*."):
        suffix = pattern[1:]
        return host.endswith(suffix) and len(host) > len(suffix)
    if pattern.startswith("."):
        return host.endswith(pattern)
    return host == pattern


def is_local_http_container_url(url: str) -> bool:
    """Check if URL points to a local HTTP container (not HTTPS)."""
    fn = getattr(synth_ai_py, "container_is_local_http_url", None)
    if callable(fn):
        return bool(fn(url))
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "http":
        return False
    return is_local_hostname(parsed.hostname)


def is_synth_managed_ngrok_url(url: str) -> bool:
    """Return True if URL is an approved Synth-managed ngrok-compatible endpoint."""
    fn = getattr(synth_ai_py, "container_is_synth_managed_ngrok_url", None)
    if callable(fn):
        return bool(fn(url))
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "https":
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    if is_cloudflare_tunnel_url(url) or is_free_ngrok_url(url):
        return False
    patterns = ["usesynth.ai", "*.usesynth.ai"]
    extra = (os.getenv("SYNTH_MANAGED_TUNNEL_TRUSTED_HOSTS") or "").strip().lower()
    if extra:
        for raw in extra.split(","):
            value = raw.strip()
            if value and value not in patterns:
                patterns.append(value)
    return any(_host_matches_pattern(hostname, pattern) for pattern in patterns)


def _is_synthtunnel_url_py(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if not parsed.path.startswith("/s/rt_"):
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    patterns = [
        "st.usesynth.ai",
        "*.st.usesynth.ai",
        "infra-api-dev.usesynth.ai",
        "infra-api.usesynth.ai",
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
