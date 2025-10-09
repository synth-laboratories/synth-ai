import os

import backoff

# Number of retry attempts that some legacy decorators rely on.
BACKOFF_TOLERANCE: int = 20

# Maximum wall-clock seconds allowed for exponential back-off when retrying an
# LLM API call.  This can be overridden at runtime with the environment
# variable `SYNTH_AI_MAX_BACKOFF`.

try:
    MAX_BACKOFF: int = max(1, int(os.getenv("SYNTH_AI_MAX_BACKOFF", "120")))
except ValueError:
    MAX_BACKOFF = 120

# Re-export backoff for convenient import patterns elsewhere
__all__ = [
    "BACKOFF_TOLERANCE",
    "MAX_BACKOFF",
    "backoff",
]
