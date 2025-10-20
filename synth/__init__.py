from synth_ai import help as _sdk_help

raise ImportError(
    "No module named 'synth'. Did you mean 'synth_ai'?\n\n"
    f"{_sdk_help()}"
)
