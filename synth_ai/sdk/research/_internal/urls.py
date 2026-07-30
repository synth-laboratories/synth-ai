"""Backend URL helpers.

Re-exports :mod:`synth_ai.core.utils.urls` rather than resolving separately.

This module used to compute its own default::

    BACKEND_URL_BASE = (os.getenv("SYNTH_BACKEND_URL") or "https://api.usesynth.ai").strip()

which read one environment variable and then hardcoded production. That bypassed
the resolution in ``core.utils.urls`` -- ``SYNTH_BACKEND_URL_OVERRIDE``, the
dev and prod chains, ``ENVIRONMENT`` -- and it was the only door onto the package
default that resolved to production rather than the local default. A caller who
set ``SYNTH_REQUIRE_EXPLICIT_BACKEND`` to guarantee no unnamed backend is
reachable still reached production through this path, because
``synth_ai.sdk.research.auth`` imports from here.

ed02f8f6 removed the equivalent hardcoded prod fallback from legacy backend
resolution; this copy survived it.
"""

from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

__all__ = ["BACKEND_URL_BASE", "normalize_backend_base"]
