"""Backend URL helpers.

Re-exports rather than reimplements. This module used to carry its own `BACKEND_URL_BASE`
built from `SYNTH_BACKEND_URL` and a hardcoded `https://api.usesynth.ai` — a second answer to
"which backend", blind to `SYNTH_BACKEND_URL_OVERRIDE`, `ENVIRONMENT`, and the guard in
`default_backend_base`. One resolver, one answer.
"""

from synth_ai.core.utils.urls import (
    BACKEND_URL_BASE,
    default_backend_base,
    normalize_backend_base,
)

__all__ = ["BACKEND_URL_BASE", "default_backend_base", "normalize_backend_base"]
