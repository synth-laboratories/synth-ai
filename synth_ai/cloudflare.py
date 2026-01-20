"""Re-exports for cloudflare integration functions.

This module provides backwards-compatible imports for cloudflare-related
functions that are now located in synth_ai.core.integrations.cloudflare.
"""

from synth_ai.core.integrations.cloudflare import (
    resolve_hostname_with_explicit_resolvers,
    verify_tunnel_dns_resolution,
)

__all__ = [
    "resolve_hostname_with_explicit_resolvers",
    "verify_tunnel_dns_resolution",
]
