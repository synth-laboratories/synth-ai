"""Shared base for SDK namespace wrappers.

Namespaces close over the public client so transport and request shaping stay in
one place (principle: sparse interconnects from SDK subtrees to the control client).

# See: Synth Style — ``specifications/tanha/references/synthstyle.md`` (backend repo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth_ai.core.research.session.client import ResearchSession


class _ClientNamespace:
    def __init__(self, client: ResearchSession) -> None:
        self._client = client


__all__ = ["_ClientNamespace"]
