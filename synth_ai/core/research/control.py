"""Research control-plane client (README smoke / ReportBench drivers).

Alpha: thin public wrapper over ``synth_ai.core.research._legacy.sdk.client.SmrControlClient``.
HTTP paths remain ``/smr`` until backend ``/research`` aliases land.
"""

from __future__ import annotations

from typing import Any

from synth_ai.core.research._legacy.sdk.client import SmrControlClient


class ResearchControlClient(SmrControlClient):
    """Canonical control-plane client for Research runs (alpha)."""

    def __init__(
        self,
        *,
        api_key: str,
        backend_base: str,
        timeout_seconds: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            backend_base=backend_base,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )


__all__ = ["ResearchControlClient"]
