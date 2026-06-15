"""Factory and Effort SDK namespaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.factories import (
    Effort,
    EffortCreateRequest,
    EffortPatchRequest,
    Factory,
    FactoryCreateRequest,
    FactoryStatus,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class FactoriesAPI(_ClientNamespace):
    def create(
        self,
        request: FactoryCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Factory:
        return Factory.from_wire(self._client.create_factory(request))

    def list(self) -> list[Factory]:
        return [Factory.from_wire(item) for item in self._client.list_factories()]

    def get(self, factory_id: str) -> Factory:
        return Factory.from_wire(self._client.get_factory(factory_id))

    def status(self, factory_id: str) -> FactoryStatus:
        return FactoryStatus.from_wire(self._client.get_factory_status(factory_id))

    def list_efforts(self, factory_id: str) -> list[Effort]:
        return [
            Effort.from_wire(item) for item in self._client.list_efforts_for_factory(factory_id)
        ]


class EffortsAPI(_ClientNamespace):
    def create(
        self,
        request: EffortCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Effort:
        return Effort.from_wire(self._client.create_effort(request))

    def get(self, effort_id: str) -> Effort:
        return Effort.from_wire(self._client.get_effort(effort_id))

    def patch(
        self,
        effort_id: str,
        request: EffortPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Effort:
        return Effort.from_wire(self._client.patch_effort(effort_id, request))


__all__ = ["EffortsAPI", "FactoriesAPI"]
