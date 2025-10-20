_HELP_TEXT = """Synth-AI SDK

Quickstart:
    pip install synth-ai
    import synth_ai
    uvx synth-ai setup

Docs → https://docs.usesynth.ai/sdk/get-started
"""

raise ImportError(
    "No module named 'synth'. Did you mean 'synth_ai'?\n\n"
    f"{_HELP_TEXT}"
)
