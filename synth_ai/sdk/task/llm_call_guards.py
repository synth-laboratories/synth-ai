"""Guards to detect and warn about direct LLM provider calls that bypass trace capture.

When LocalAPIs call LLM providers directly (api.openai.com, api.anthropic.com, etc.)
instead of using the inference_url from policy_config, traces aren't captured by
the interceptor, breaking trace hydration.

This module provides runtime guards that detect such calls and warn developers.
"""

import warnings
from typing import Set
from urllib.parse import urlparse


# Known LLM provider domains that should be accessed via interceptor
LLM_PROVIDER_DOMAINS: Set[str] = {
    "api.openai.com",
    "api.anthropic.com",
    "api.groq.com",
    "generativelanguage.googleapis.com",  # Google Gemini
    "api.cohere.ai",
    "api.together.xyz",
    "api.perplexity.ai",
}


def check_url_for_direct_provider_call(url: str) -> bool:
    """Check if a URL is a direct call to an LLM provider.

    Args:
        url: The URL being called

    Returns:
        True if this is a direct provider call (should warn), False otherwise
    """
    try:
        parsed = urlparse(str(url))
        hostname = parsed.hostname or ""

        # Check against known provider domains
        for provider_domain in LLM_PROVIDER_DOMAINS:
            if hostname == provider_domain or hostname.endswith(f".{provider_domain}"):
                return True

        return False
    except Exception:
        return False


def warn_if_direct_provider_call(url: str, stacklevel: int = 2) -> None:
    """Warn if URL is a direct call to an LLM provider.

    Args:
        url: The URL being called
        stacklevel: Stack level for warning (default: 2)
    """
    if check_url_for_direct_provider_call(url):
        warnings.warn(
            f"Direct call to LLM provider detected: {url}\n"
            f"This bypasses trace capture by the Synth AI interceptor.\n"
            f"Use inference_url from policy_config instead.\n"
            f"See: https://docs.usesynth.ai/guides/local-api#inference-url",
            UserWarning,
            stacklevel=stacklevel
        )


def install_httpx_guard() -> None:
    """Install guard on httpx.AsyncClient to detect direct provider calls.

    This monkey-patches httpx.AsyncClient.post to warn when URLs point
    directly to LLM providers instead of through the interceptor.

    Call this once at module initialization in your LocalAPI.
    """
    try:
        import httpx

        # Check if already patched
        if hasattr(httpx.AsyncClient.post, '_synth_guarded'):
            return

        _original_post = httpx.AsyncClient.post

        async def _guarded_post(self, url, *args, **kwargs):
            warn_if_direct_provider_call(url, stacklevel=3)
            return await _original_post(self, url, *args, **kwargs)

        _guarded_post._synth_guarded = True  # type: ignore
        httpx.AsyncClient.post = _guarded_post  # type: ignore

    except ImportError:
        # httpx not installed, skip
        pass


def install_requests_guard() -> None:
    """Install guard on requests to detect direct provider calls.

    This monkey-patches requests.post to warn when URLs point directly
    to LLM providers instead of through the interceptor.

    Call this once at module initialization in your LocalAPI.
    """
    try:
        import requests

        # Check if already patched
        if hasattr(requests.post, '_synth_guarded'):
            return

        _original_post = requests.post

        def _guarded_post(url, *args, **kwargs):
            warn_if_direct_provider_call(url, stacklevel=3)
            return _original_post(url, *args, **kwargs)

        _guarded_post._synth_guarded = True  # type: ignore
        requests.post = _guarded_post  # type: ignore

    except ImportError:
        # requests not installed, skip
        pass


def install_openai_guard() -> None:
    """Install guard on OpenAI client to detect direct instantiation.

    This warns when OpenAI client is instantiated directly instead of
    using the inference_url pattern.

    Call this once at module initialization in your LocalAPI.
    """
    try:
        import openai

        # Check if already patched
        if hasattr(openai.OpenAI.__init__, '_synth_guarded'):
            return

        _original_init = openai.OpenAI.__init__

        def _guarded_init(self, *args, **kwargs):
            warnings.warn(
                "Direct OpenAI client instantiation detected.\n"
                "For proper trace capture, use inference_url from policy_config with httpx instead.\n"
                "See: https://docs.usesynth.ai/guides/local-api#llm-calls",
                UserWarning,
                stacklevel=3
            )
            return _original_init(self, *args, **kwargs)

        _guarded_init._synth_guarded = True  # type: ignore
        openai.OpenAI.__init__ = _guarded_init  # type: ignore

    except ImportError:
        # openai not installed, skip
        pass


def install_anthropic_guard() -> None:
    """Install guard on Anthropic client to detect direct instantiation.

    This warns when Anthropic client is instantiated directly instead of
    using the inference_url pattern.

    Call this once at module initialization in your LocalAPI.
    """
    try:
        import anthropic

        # Check if already patched
        if hasattr(anthropic.Anthropic.__init__, '_synth_guarded'):
            return

        _original_init = anthropic.Anthropic.__init__

        def _guarded_init(self, *args, **kwargs):
            warnings.warn(
                "Direct Anthropic client instantiation detected.\n"
                "For proper trace capture, use inference_url from policy_config with httpx instead.\n"
                "See: https://docs.usesynth.ai/guides/local-api#llm-calls",
                UserWarning,
                stacklevel=3
            )
            return _original_init(self, *args, **kwargs)

        _guarded_init._synth_guarded = True  # type: ignore
        anthropic.Anthropic.__init__ = _guarded_init  # type: ignore

    except ImportError:
        # anthropic not installed, skip
        pass


def install_all_guards() -> None:
    """Install all available guards for detecting direct LLM calls.

    This is a convenience function that installs guards on:
    - httpx.AsyncClient.post (for direct HTTP calls)
    - requests.post (for direct HTTP calls)
    - openai.OpenAI.__init__ (for direct OpenAI client usage)
    - anthropic.Anthropic.__init__ (for direct Anthropic client usage)

    Usage:
        from synth_ai.sdk.task.llm_call_guards import install_all_guards
        install_all_guards()

    Safe to call multiple times - guards won't be installed twice.
    """
    install_httpx_guard()
    install_requests_guard()
    install_openai_guard()
    install_anthropic_guard()
