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
    _env_or_default("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
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
    _env_or_default("SYNTH_RUST_BACKEND_URL", "https://infra-api.usesynth.ai"),
)


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
        hostname = urlparse(url).hostname or ""
    except Exception:
        return False
    if hostname == "st.usesynth.ai":
        return True
    return hostname.endswith(".st.usesynth.ai")
