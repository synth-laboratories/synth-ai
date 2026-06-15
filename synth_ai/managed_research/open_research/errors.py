"""Typed error envelope for Open Research v1.

The HTTP contract defines a fixed set of error ``class`` values. The MCP
caller branches on the class, so we lift it onto a typed exception and
keep the ``actionable`` text exactly as the backend returned it (public-
safe, never re-worded by the SDK).
"""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.errors import SmrApiError

OPEN_RESEARCH_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "auth",
        "quota",
        "config",
        "safety",
        "validity",
        "novelty",
        "theme_fit",
        "transient",
        "not_found",
    }
)


class OpenResearchError(SmrApiError):
    """Raised when the Open Research backend returns a typed error envelope.

    Mirrors:

    .. code-block:: json

        {"error": {"class": "...", "code": "...", "message": "...",
                   "actionable": "...", "retry_after_seconds": null,
                   "request_id": "..."}}
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        code: str,
        actionable: str,
        status_code: int | None = None,
        retry_after_seconds: int | None = None,
        request_id: str | None = None,
        response_text: str | None = None,
        envelope: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
        )
        self.error_class = error_class
        self.code = code
        self.actionable = actionable
        self.retry_after_seconds = retry_after_seconds
        self.request_id = request_id
        self.envelope: dict[str, Any] = dict(envelope) if envelope else {}

    def to_mcp_payload(self) -> dict[str, Any]:
        """Return the public-safe error structure shipped to MCP callers."""
        payload: dict[str, Any] = {
            "error": {
                "class": self.error_class,
                "code": self.code,
                "message": str(self),
                "actionable": self.actionable,
                "retry_after_seconds": self.retry_after_seconds,
                "request_id": self.request_id,
            }
        }
        if self.status_code is not None:
            payload["http_status"] = self.status_code
        return payload


def parse_open_research_error_envelope(
    payload: Any,
    *,
    status_code: int | None,
    response_text: str | None,
) -> OpenResearchError | None:
    """Extract an :class:`OpenResearchError` from a contract-shaped body.

    Returns ``None`` when the body is not a recognized envelope; callers
    fall back to the generic :class:`SmrApiError` mapping in that case.
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    raw_class = error.get("class")
    raw_code = error.get("code")
    if not isinstance(raw_class, str) or not isinstance(raw_code, str):
        return None
    if raw_class not in OPEN_RESEARCH_ERROR_CLASSES:
        return None
    message_raw = error.get("message")
    actionable_raw = error.get("actionable")
    message = message_raw.strip() if isinstance(message_raw, str) else ""
    actionable = actionable_raw.strip() if isinstance(actionable_raw, str) else ""
    retry_after = error.get("retry_after_seconds")
    if not isinstance(retry_after, int):
        retry_after = None
    request_id_raw = error.get("request_id")
    request_id = request_id_raw if isinstance(request_id_raw, str) and request_id_raw else None
    return OpenResearchError(
        message or f"Open Research backend rejected request ({raw_class})",
        error_class=raw_class,
        code=raw_code,
        actionable=actionable,
        status_code=status_code,
        retry_after_seconds=retry_after,
        request_id=request_id,
        response_text=response_text,
        envelope=payload,
    )


__all__ = [
    "OPEN_RESEARCH_ERROR_CLASSES",
    "OpenResearchError",
    "parse_open_research_error_envelope",
]
