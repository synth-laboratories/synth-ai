"""Guards to detect and warn about direct LLM provider calls that bypass trace capture."""

import warnings

import synth_ai_py


def check_url_for_direct_provider_call(url: str) -> bool:
    """Check if a URL is a direct call to an LLM provider."""

    return synth_ai_py.localapi_check_url_for_direct_provider_call(str(url))


def warn_if_direct_provider_call(url: str, stacklevel: int = 2) -> None:
    """Warn if URL is a direct call to an LLM provider."""

    if check_url_for_direct_provider_call(url):
        warnings.warn(
            f"Direct call to LLM provider detected: {url}\n"
            f"This bypasses trace capture by the Synth AI interceptor.\n"
            f"Use inference_url from policy_config instead.\n"
            f"See: https://docs.usesynth.ai/guides/local-api#inference-url",
            UserWarning,
            stacklevel=stacklevel,
        )


def install_httpx_guard() -> None:
    """Install guard on httpx.AsyncClient to detect direct provider calls."""
    try:
        import importlib

        httpx = importlib.import_module("httpx")

        if hasattr(httpx.AsyncClient.post, "_synth_guarded"):
            return

        _original_post = httpx.AsyncClient.post

        async def _guarded_post(self, url, *args, **kwargs):
            warn_if_direct_provider_call(url, stacklevel=3)
            return await _original_post(self, url, *args, **kwargs)

        _guarded_post._synth_guarded = True  # type: ignore
        httpx.AsyncClient.post = _guarded_post  # type: ignore
    except ImportError:
        pass


def install_requests_guard() -> None:
    """Install guard on requests to detect direct provider calls."""
    try:
        import requests

        if hasattr(requests.post, "_synth_guarded"):
            return

        _original_post = requests.post

        def _guarded_post(url, *args, **kwargs):
            warn_if_direct_provider_call(url, stacklevel=3)
            return _original_post(url, *args, **kwargs)

        _guarded_post._synth_guarded = True  # type: ignore
        requests.post = _guarded_post  # type: ignore
    except ImportError:
        pass


def install_openai_guard() -> None:
    """Install guard on OpenAI client to detect direct instantiation."""
    try:
        import openai

        if hasattr(openai.OpenAI.__init__, "_synth_guarded"):
            return

        _original_init = openai.OpenAI.__init__

        def _guarded_init(self, *args, **kwargs):
            warnings.warn(
                "Direct OpenAI client instantiation detected.\n"
                "For proper trace capture, use inference_url from policy_config with httpx instead.\n"
                "See: https://docs.usesynth.ai/guides/local-api#llm-calls",
                UserWarning,
                stacklevel=3,
            )
            return _original_init(self, *args, **kwargs)

        _guarded_init._synth_guarded = True  # type: ignore
        openai.OpenAI.__init__ = _guarded_init  # type: ignore
    except ImportError:
        pass


def install_anthropic_guard() -> None:
    """Install guard on Anthropic client to detect direct instantiation."""
    try:
        import anthropic

        if hasattr(anthropic.Anthropic.__init__, "_synth_guarded"):
            return

        _original_init = anthropic.Anthropic.__init__

        def _guarded_init(self, *args, **kwargs):
            warnings.warn(
                "Direct Anthropic client instantiation detected.\n"
                "For proper trace capture, use inference_url from policy_config with httpx instead.\n"
                "See: https://docs.usesynth.ai/guides/local-api#llm-calls",
                UserWarning,
                stacklevel=3,
            )
            return _original_init(self, *args, **kwargs)

        _guarded_init._synth_guarded = True  # type: ignore
        anthropic.Anthropic.__init__ = _guarded_init  # type: ignore
    except ImportError:
        pass


def install_all_guards() -> None:
    """Install all available guards for detecting direct LLM calls."""
    install_httpx_guard()
    install_requests_guard()
    install_openai_guard()
    install_anthropic_guard()
