"""Drop-in `gepa` module shim provided by prompt-opt."""

from prompt_opt.gepa_ai_compat import LocalGEPAAdapterProtocol, optimize

__all__ = ["LocalGEPAAdapterProtocol", "optimize"]
