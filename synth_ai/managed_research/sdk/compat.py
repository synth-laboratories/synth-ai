"""Compatibility surfaces for older SDK entrypoints.

The optional ``synth-ai`` bridge is loaded only when callers use OpenAI Agents SDK
helpers; ``ImportError`` is translated to a single actionable ``RuntimeError``.

# See: Synth Style — compatibility layers isolated; never hide causes (``from exc``).
"""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk.config import (
    OPENAI_TRANSPORT_MODE_AUTO,
    optional_str,
    resolve_openai_transport_mode,
)


class SmrControlClientMixin:
    """Optional synth-ai bridge retained for one compatibility window."""

    openai_transport_mode: str
    openai_organization: str | None
    openai_project: str | None
    openai_request_id: str | None
    _synth_client: Any | None

    def _initialize_openai_bridge(
        self,
        *,
        openai_transport_mode: str,
        openai_organization: str | None,
        openai_project: str | None,
        openai_request_id: str | None,
    ) -> None:
        self.openai_transport_mode = resolve_openai_transport_mode(openai_transport_mode)
        self.openai_organization = optional_str(openai_organization)
        self.openai_project = optional_str(openai_project)
        self.openai_request_id = optional_str(openai_request_id)
        self._synth_client = None

    def close_openai_bridge(self) -> None:
        synth_client = getattr(self, "_synth_client", None)
        if synth_client is not None:
            close_fn = getattr(synth_client, "close", None)
            if callable(close_fn):
                close_fn()
        self._synth_client = None

    def _build_synth_client(
        self,
        *,
        openai_transport_mode: str,
        openai_organization: str | None,
        openai_project: str | None,
        openai_request_id: str | None,
    ) -> Any:
        try:
            from synth_ai import SynthClient
        except (
            ImportError,
            ModuleNotFoundError,
        ) as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "synth-ai is required for OpenAI Agents SDK access from managed-research. "
                "Install it with `uv add synth-ai`."
            ) from exc

        return SynthClient(
            api_key=self.api_key,
            base_url=self.backend_base,
            timeout=self.timeout_seconds,
            openai_transport_mode=openai_transport_mode,
            openai_organization=openai_organization,
            openai_project=openai_project,
            openai_request_id=openai_request_id,
        )

    @property
    def synth_ai(self) -> Any:
        if self._synth_client is None:
            self._synth_client = self._build_synth_client(
                openai_transport_mode=self.openai_transport_mode,
                openai_organization=self.openai_organization,
                openai_project=self.openai_project,
                openai_request_id=self.openai_request_id,
            )
        return self._synth_client

    @property
    def openai_agents_sdk(self) -> Any:
        return self.synth_ai.openai_agents_sdk

    @property
    def managed_agents(self) -> Any:
        return self.synth_ai.managed_agents

    def openai_agents_sdk_client(
        self,
        *,
        transport_mode: str | None = None,
        openai_organization: str | None = None,
        openai_project: str | None = None,
        request_id: str | None = None,
    ) -> Any:
        if (
            transport_mode is None
            and openai_organization is None
            and openai_project is None
            and request_id is None
        ):
            return self.openai_agents_sdk

        return self._build_synth_client(
            openai_transport_mode=resolve_openai_transport_mode(
                transport_mode or self.openai_transport_mode
            ),
            openai_organization=optional_str(openai_organization or self.openai_organization),
            openai_project=optional_str(openai_project or self.openai_project),
            openai_request_id=optional_str(request_id or self.openai_request_id),
        ).openai_agents_sdk


__all__ = [
    "OPENAI_TRANSPORT_MODE_AUTO",
    "SmrControlClientMixin",
]
