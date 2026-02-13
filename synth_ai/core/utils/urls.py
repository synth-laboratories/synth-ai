from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for URL utilities.") from exc


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
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        return False
    if "/s/rt_" in (parsed.path or ""):
        return True
    if hostname == "st.usesynth.ai":
        return True
    return hostname.endswith(".st.usesynth.ai")
