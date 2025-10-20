from importlib import import_module

_HELP_FALLBACK = """Synth-AI SDK

Quickstart:
    pip install synth-ai
    import synth_ai
    uvx synth-ai setup

Docs → https://docs.usesynth.ai/sdk/get-started
"""


def _load_help() -> str:
    try:
        synth_ai = import_module("synth_ai")
        helper = getattr(synth_ai, "help", None)
        text = helper() if callable(helper) else None
    except Exception:
        return _HELP_FALLBACK
    return text or _HELP_FALLBACK


_HELP_TEXT = _load_help()
__doc__ = _HELP_TEXT


def help() -> str:
    """Return a concise quickstart for the Synth-AI SDK."""
    return _HELP_TEXT
