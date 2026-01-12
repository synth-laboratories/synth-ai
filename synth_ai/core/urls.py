import os

# Base URL for all backends
BACKEND_URL_BASE = os.getenv("SYNTH_BACKEND_URL") or "https://api.usesynth.ai"

# API URL (base + /api suffix) for endpoints that expect this format
BACKEND_URL_API = BACKEND_URL_BASE + "/api"

# Synth Research API base (supports OpenAI, Anthropic, and custom formats)
# Real routes: /api/synth-research/chat/completions, /api/synth-research/messages
# V1 routes: /api/synth-research/v1/chat/completions, /api/synth-research/v1/messages
BACKEND_URL_SYNTH_RESEARCH_BASE = BACKEND_URL_BASE + "/api/synth-research"

# Provider-specific URLs (for SDKs that expect standard paths)
BACKEND_URL_SYNTH_RESEARCH_OPENAI = (
    BACKEND_URL_SYNTH_RESEARCH_BASE + "/v1"
)  # For OpenAI SDKs (appends /chat/completions)
BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC = (
    BACKEND_URL_SYNTH_RESEARCH_BASE  # For Anthropic SDKs (appends /v1/messages)
)


FRONTEND_URL_BASE = os.getenv("SYNTH_FRONTEND_URL") or "https://usesynth.ai"


def join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not path:
        return base
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def normalize_base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[: -len("/api")]
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized


def local_backend_url(host: str = "localhost", port: int = 8000) -> str:
    return f"http://{host}:{port}"


def backend_health_url(base_url: str) -> str:
    return join_url(base_url, "/health")


def backend_me_url(base_url: str) -> str:
    return join_url(base_url, "/api/v1/me")


def backend_demo_keys_url(base_url: str) -> str:
    return join_url(base_url, "/api/demo/keys")
