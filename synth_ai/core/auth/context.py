"""Typed request authority context.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass


class OrganizationId(str):
    """Opaque Synth organization identifier."""


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Optional tenant and correlation context applied to one operation."""

    organization_id: OrganizationId | None = None
    request_id: str | None = None


__all__ = ["OrganizationId", "RequestContext"]
