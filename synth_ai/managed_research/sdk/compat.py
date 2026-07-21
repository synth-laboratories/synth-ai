"""Retired compatibility surfaces for older managed-agents entrypoints."""

from __future__ import annotations


class SmrControlClientMixin:
    """Fail loudly for bridges whose backend policy boundary was retired."""

    @staticmethod
    def _retired_bridge_error(surface: str) -> RuntimeError:
        return RuntimeError(
            f"{surface} was retired with the backend managed-agents proxy. "
            "Use the explicit synth-ai Horizons Private client only when you have a "
            "Horizons Private base URL and credential; a Synth API key is not a "
            "replacement service credential."
        )

    def close_openai_bridge(self) -> None:
        """Retained as a no-op for callers that close the legacy bridge."""

    @property
    def synth_ai(self) -> object:
        raise self._retired_bridge_error("SmrControlClient.synth_ai")

    @property
    def openai_agents_sdk(self) -> object:
        raise self._retired_bridge_error("SmrControlClient.openai_agents_sdk")

    @property
    def managed_agents(self) -> object:
        raise self._retired_bridge_error("SmrControlClient.managed_agents")

    def openai_agents_sdk_client(
        self,
        *,
        transport_mode: str | None = None,
        openai_organization: str | None = None,
        openai_project: str | None = None,
        request_id: str | None = None,
    ) -> object:
        del transport_mode, openai_organization, openai_project, request_id
        raise self._retired_bridge_error("SmrControlClient.openai_agents_sdk_client()")


__all__ = [
    "SmrControlClientMixin",
]
