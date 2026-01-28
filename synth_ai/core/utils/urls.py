try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for URL utilities.") from exc

BACKEND_URL_BASE = synth_ai_py.backend_url_base()
BACKEND_URL_API = synth_ai_py.backend_url_api()
BACKEND_URL_SYNTH_RESEARCH_BASE = synth_ai_py.backend_url_synth_research_base()
BACKEND_URL_SYNTH_RESEARCH_OPENAI = synth_ai_py.backend_url_synth_research_openai()
BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC = synth_ai_py.backend_url_synth_research_anthropic()
FRONTEND_URL_BASE = synth_ai_py.frontend_url_base()


def join_url(base_url: str, path: str) -> str:
    return synth_ai_py.join_url(base_url, path)


def normalize_base_url(url: str) -> str:
    return synth_ai_py.normalize_backend_base(url)


def normalize_backend_base(url: str) -> str:
    """Normalize backend base URL via the Rust core when available."""
    return synth_ai_py.normalize_backend_base(url)


def normalize_inference_base(url: str) -> str:
    """Normalize inference base URL via the Rust core when available."""
    return synth_ai_py.normalize_inference_base(url)


def local_backend_url(host: str = "localhost", port: int = 8000) -> str:
    return synth_ai_py.local_backend_url(host, port)


def backend_health_url(base_url: str) -> str:
    return synth_ai_py.backend_health_url(base_url)


def backend_me_url(base_url: str) -> str:
    return synth_ai_py.backend_me_url(base_url)


def backend_demo_keys_url(base_url: str) -> str:
    return synth_ai_py.backend_demo_keys_url(base_url)
