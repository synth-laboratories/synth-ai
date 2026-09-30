"""Explain a 403 by the scopes it needed and the scopes the token held.

Only facts the server or the SDK scope table state are reported. A granted list
the server did not send is reported as unknown, never guessed. A 403 that is not
about scope (role, assignment, ownership, organization) says so instead of
inviting a useless re-consent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from synth_ai.core.errors import SynthError, SynthErrorCategory

from .scopes import OPERATION_SCOPES

INSUFFICIENT_SCOPE = "insufficient_scope"
_NESTING_KEYS = ("detail", "data", "error")


@dataclass(frozen=True, slots=True)
class ScopeDenial:
    """Reported authorization facts without inventing token grants.

    Attributes:
        insufficient_scope: Whether the server identified a scope failure.
        required_any_of: Any-of required scopes reported by the server or SDK table.
        required_source: Origin of required scopes: server, sdk_table or unknown.
        granted: Reported token scopes, or None when the server did not disclose them.
        operation: SDK operation identifier, or None when unavailable.

    Examples:
        ```python
        denial = ScopeDenial(True, ("index:review",), "server", (), "index.qa.reviews.record")
        print(denial.message())
        ```
    """

    insufficient_scope: bool
    required_any_of: tuple[str, ...]
    required_source: str  # "server", "sdk_table" or "unknown"
    granted: tuple[str, ...] | None
    operation: str | None

    def message(self) -> str:
        """Format known scope facts and an appropriate recovery hint.

        Returns:
            A sanitized explanation retaining unknown granted scopes as unknown.
        """
        required = " or ".join(self.required_any_of) if self.required_any_of else "unknown"
        granted = (
            "unknown (the server did not report them)"
            if self.granted is None
            else (", ".join(self.granted) or "none")
        )
        origin = {"server": "server", "sdk_table": "SDK scope table"}.get(self.required_source)
        needed = f"{required} ({origin})" if origin else required
        head = (
            "insufficient_scope"
            if self.insufficient_scope
            else "forbidden (not reported as a scope failure)"
        )
        text = f"{head}: needs any of {needed}; token has {granted}."
        return text + " " + self.hint()

    def hint(self) -> str:
        """Explain the authorization boundary relevant to this denial.

        Returns:
            Scope-consent guidance for a scope failure, otherwise role/assignment guidance.
        """
        if self.insufficient_scope:
            return (
                "Reconnect and approve the missing scope. Scopes only allow a class of "
                "operation; even with them the backend decides by role, assignment and "
                "ownership."
            )
        return (
            "The scope class may already be sufficient: the backend also refuses by role, "
            "assignment (accepted, unexpired, unrevoked), ownership or organization. "
            "Check that you act as the right party for this case."
        )


def _strings(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        return tuple(value.split())
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    return None


def _walk(value: Any, depth: int = 0) -> dict[str, Any]:
    """Collect scope facts from a (possibly nested) error body."""
    found: dict[str, Any] = {}
    if depth > 3 or not isinstance(value, dict):
        return found
    for key in ("required_scopes", "granted_scopes"):
        strings = _strings(value.get(key))
        if strings is not None:
            found[key] = strings
    for key in ("oauth_error", "error", "code", "error_code"):
        if value.get(key) == INSUFFICIENT_SCOPE:
            found["insufficient"] = True
    for key in _NESTING_KEYS:
        for name, item in _walk(value.get(key), depth + 1).items():
            found.setdefault(name, item)
    return found


def scope_denial(error: BaseException) -> ScopeDenial | None:
    """Extract scope facts from an SDK authorization failure.

    Args:
        error: Caught exception to inspect; unrelated exceptions are not rewritten.

    Returns:
        ScopeDenial for an authorization-category SynthError, otherwise None.
        Missing granted scopes remain unknown; server requirements take precedence.
    """
    if not isinstance(error, SynthError) or error.failure is None:
        return None
    failure = error.failure
    if failure.category is not SynthErrorCategory.AUTHORIZATION:
        return None
    facts: dict[str, Any] = {}
    detail = getattr(error, "detail", None)
    facts.update(_walk(detail))
    snippet = getattr(error, "body_snippet", None)
    if isinstance(snippet, str):
        try:
            for name, item in _walk(json.loads(snippet)).items():
                facts.setdefault(name, item)
        except ValueError:
            pass
    insufficient = bool(facts.get("insufficient")) or str(failure.code) == INSUFFICIENT_SCOPE
    required = facts.get("required_scopes")
    source = "server"
    if required is None:
        table = OPERATION_SCOPES.get(failure.operation or "")
        required, source = (table, "sdk_table") if table else ((), "unknown")
    return ScopeDenial(
        insufficient_scope=insufficient,
        required_any_of=tuple(required),
        required_source=source,
        granted=facts.get("granted_scopes"),
        operation=failure.operation,
    )
