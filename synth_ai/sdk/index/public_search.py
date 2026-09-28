"""Index Search v0.2 public contract: credential-optional Fast and Deep search.

``POST /api/v1/index/public/search`` works with no Authorization header and with
an API key. Fast answers synchronously (200); Deep is admitted (202) and polled
at ``GET /api/v1/index/public/searches/{search_id}`` with the per-search token in
``X-Search-Token``. The token is the only credential to the result: it lives on
the handle, is never part of a result object, repr, log line or file.

The backend owns price, limits, retention and privacy wording; read them from
``Capabilities.public_search`` and render them with :func:`public_search_copy`.
Nothing here hardcodes a price. The adapter is deliberately thin so a backend
field rename is a one-line fix in ``_parse_result`` / ``_parse_item``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from synth_ai.core.errors import HTTPError, SynthError, SynthFailure
from synth_ai.core.http.transport import HttpTransport

from .catalog import PublicSearchCapability
from .errors import IndexErrorCode
from .search import SearchMode

PUBLIC_SEARCH_TOKEN_HEADER = "X-Search-Token"
#: Monitor release decision id. Delivered as a response header (never in the body) so
#: the reviewed bytes equal the delivered bytes; the body's ``monitor.release_id`` is null.
PUBLIC_SEARCH_RELEASE_HEADER = "X-Index-Monitor-Release"
#: Public-only delivery fields travel in headers outside the Monitor-reviewed body
#: (backend #1660): the body is exactly the Monitor's public contract key set.
PUBLIC_SEARCH_ID_HEADER = "X-Index-Search-Id"
PUBLIC_SEARCH_TOKEN_EXPIRES_HEADER = "X-Search-Token-Expires-At"
PUBLIC_SEARCH_CHARGE_HEADER = "X-Index-Customer-Charge-Cents"
PUBLIC_SEARCH_INTERNAL_COST_HEADER = "X-Index-Internal-Cost-Recorded"
#: Monitor delivery marker, delivered next to the release header.
PUBLIC_SEARCH_DELIVERY_HEADER = "X-Index-Monitor-Delivery"
#: Poll cadence when the backend sends no ``Retry-After`` on a 202.
DEFAULT_PUBLIC_POLL_SECONDS = 1.0
MIN_PUBLIC_POLL_SECONDS = 0.05
MAX_PUBLIC_POLL_SECONDS = 30.0
#: One status read is abandoned and retried after this long (see client.py
#: STATUS_POLL_TIMEOUT_SECONDS for the incident that set it).
PUBLIC_POLL_REQUEST_TIMEOUT_SECONDS = 20.0
#: Deep public search waits this long by default (backend Deep deadline + slack).
DEFAULT_PUBLIC_SEARCH_WAIT_SECONDS = 185.0

_P = "/api/v1/index/public"
_SEARCH_PATH = f"{_P}/search"
_SEARCH_ITEM_PATH = f"{_P}/searches/{{search_id}}"
_SEARCH_CANCEL_PATH = f"{_P}/searches/{{search_id}}/cancel"


# Results ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicSearchCitation:
    """One cited Contribution revision (backend ``ContributionReference``).

    The delivery carries no titles or excerpts: ``response`` cites the
    ``contribution_id`` inline as ``[<contribution_id>]`` and callers fetch the
    revision by id when they need its content.
    """

    contribution_id: str
    revision_id: str
    raw: Mapping[str, Any]

    @property
    def citation(self) -> str:
        """Exact revision label, ``<contribution_id>@<revision_id>``."""
        return f"{self.contribution_id}@{self.revision_id}"


@dataclass(frozen=True, slots=True)
class PublicSearchResult:
    """Delivered public search envelope. Carries no search token."""

    search_id: str
    mode: SearchMode
    status: str
    #: Backend ``citations`` in first-appearance order (``results`` is an alias).
    citations: tuple[PublicSearchCitation, ...]
    customer_charge_cents: int
    monitor_release_id: str | None
    #: Text whose claims cite contribution ids inline (``[<contribution_id>]``).
    response: str
    partial_reason: str | None
    raw: Mapping[str, Any]

    @property
    def results(self) -> tuple[PublicSearchCitation, ...]:
        return self.citations


@dataclass(frozen=True, slots=True)
class PublicSearchStatus:
    """A Deep search that is admitted but not yet delivered."""

    search_id: str
    state: str
    poll_after_s: float


# Errors ----------------------------------------------------------------------


class PublicSearchError(SynthError):
    """Typed public search failure; the transport error is chained as ``__cause__``."""

    def __init__(
        self,
        message: str,
        *,
        failure: SynthFailure | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message, failure=failure)
        self.status = status

    @property
    def code(self) -> str | None:
        return str(self.failure.code) if self.failure is not None else None


class PublicSearchRateLimitedError(PublicSearchError):
    """429 ``index_public_rate_limited``; ``retry_after_s`` comes from Retry-After."""

    def __init__(
        self,
        message: str,
        *,
        scope: str | None,
        retry_after_s: float | None,
        failure: SynthFailure | None = None,
        status: int | None = 429,
    ) -> None:
        super().__init__(message, failure=failure, status=status)
        self.scope = scope
        self.retry_after_s = retry_after_s


class PublicSearchUnavailableError(PublicSearchError):
    """503: the route failed closed and produced no result."""


class PublicSearchBudgetScope(StrEnum):
    """Which public budget ran out (backend ``BudgetScope``)."""

    DAILY_CENTS = "daily_cents"
    DEEP_CONCURRENCY = "deep_concurrency"

    @classmethod
    def parse(cls, value: object) -> PublicSearchBudgetScope | None:
        """The known scope, or None for a missing or not-yet-known value (never raises)."""
        if not isinstance(value, str):
            return None
        try:
            return cls(value)
        except ValueError:
            return None


class PublicSearchBudgetExhaustedError(PublicSearchUnavailableError):
    """503 ``index_public_budget_exhausted``; ``scope`` names the exhausted budget."""

    def __init__(
        self,
        message: str,
        *,
        scope: PublicSearchBudgetScope | None = None,
        retry_after_s: float | None = None,
        failure: SynthFailure | None = None,
        status: int | None = 503,
    ) -> None:
        super().__init__(message, failure=failure, status=status)
        self.scope = scope
        self.retry_after_s = retry_after_s


class PublicSearchRateStoreUnavailableError(PublicSearchUnavailableError):
    """503 ``index_rate_store_unavailable``."""


class PublicSearchMonitorUnavailableError(PublicSearchUnavailableError):
    """503 ``monitor_unavailable``."""


class PublicSearchRequestTooLargeError(PublicSearchError):
    """413 ``index_request_too_large``."""


class PublicSearchDisabledError(PublicSearchError):
    """404 ``index_public_search_disabled``: the backend flag is off."""


class PublicSearchAuthenticatedError(PublicSearchError):
    """409 ``index_public_search_authenticated``: the free public route is anonymous-only.

    Any credential (API key, Clerk session) is refused. Keyed callers use the paid
    keyed route, ``IndexAPI.search(...)`` (MCP: ``index_private_search``).
    """


PUBLIC_SEARCH_AUTHENTICATED_MESSAGE = (
    "Public Index search is anonymous-only and refuses credentialed requests; "
    "with an API key use the paid keyed search, IndexAPI.search(...) "
    "(MCP: index_search with a key, once your organization turns on wallet payments)."
)


class PublicSearchNotFoundError(PublicSearchError):
    """404 ``index_search_not_found``: unknown search id or wrong token."""


class PublicSearchNotReadyError(PublicSearchError):
    """A replay asked for a result the backend has not delivered yet."""


class PublicSearchFailedError(PublicSearchError):
    """The backend recorded a terminal non-delivery state (failed or cancelled).

    ``failure_code``/``failure_retryable`` mirror the backend ``SearchFailure`` when present.
    """

    def __init__(
        self,
        search_id: str,
        state: str,
        *,
        failure_code: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        cause = "" if failure_code is None else f" ({failure_code})"
        super().__init__(f"Public search {search_id} ended in state {state!r}{cause}")
        self.search_id = search_id
        self.state = state
        self.failure_code = failure_code
        self.failure_retryable = retryable


class PublicSearchCancelledError(PublicSearchFailedError):
    """The search was cancelled before a result was delivered."""


class PublicSearchWaitTimeoutError(PublicSearchError):
    """The local wait ended; the handle stays valid for further polling."""

    def __init__(self, handle: PublicSearchHandle) -> None:
        super().__init__(f"Public search {handle.search_id} is still running")
        self.handle = handle
        self.search_id = handle.search_id


_UNAVAILABLE_BY_CODE: dict[str, type[PublicSearchUnavailableError]] = {
    IndexErrorCode.RATE_STORE_UNAVAILABLE: PublicSearchRateStoreUnavailableError,
    IndexErrorCode.MONITOR_UNAVAILABLE: PublicSearchMonitorUnavailableError,
}


def _error_sources(error: HTTPError) -> tuple[Mapping[str, Any], ...]:
    detail = error.detail
    if not isinstance(detail, Mapping):
        return ()
    nested = detail.get("detail")
    return tuple(item for item in (nested, detail) if isinstance(item, Mapping))


def _error_scope(error: HTTPError) -> str | None:
    return next(
        (
            value
            for source in _error_sources(error)
            if isinstance((value := source.get("scope")), str) and value
        ),
        None,
    )


def translate_public_search_error(error: SynthError) -> SynthError:
    """Map a transport error on a public search route to its typed exception.

    Non-HTTP failures (timeouts, network) are returned unchanged. Unknown codes
    become a plain :class:`PublicSearchError` and keep the raw code.
    """
    if not isinstance(error, HTTPError):
        return error
    code = str(error.error_code) if error.error_code is not None else None
    status = error.status
    failure = error.failure
    scope = _error_scope(error)
    if status == 429 or code == IndexErrorCode.PUBLIC_RATE_LIMITED:
        retry_after = error.retry_after_seconds
        wait = "" if retry_after is None else f"; retry in {retry_after:g} s"
        where = "" if scope is None else f" (scope={scope})"
        return PublicSearchRateLimitedError(
            f"Public Index search rate limited{where}{wait}",
            scope=scope,
            retry_after_s=retry_after,
            failure=failure,
            status=status,
        )
    if code == IndexErrorCode.PUBLIC_BUDGET_EXHAUSTED:
        budget_scope = PublicSearchBudgetScope.parse(scope)
        where = "" if budget_scope is None else f" (scope={budget_scope.value})"
        return PublicSearchBudgetExhaustedError(
            f"Public Index search is unavailable ({code}){where}; no result was produced "
            "and nothing was charged.",
            scope=budget_scope,
            retry_after_s=error.retry_after_seconds,
            failure=failure,
            status=status,
        )
    unavailable = _UNAVAILABLE_BY_CODE.get(code or "")
    if unavailable is not None:
        return unavailable(
            f"Public Index search is unavailable ({code}); no result was produced "
            "and nothing was charged.",
            failure=failure,
            status=status,
        )
    if code == IndexErrorCode.REQUEST_TOO_LARGE or status == 413:
        return PublicSearchRequestTooLargeError(
            "Public Index search request is too large", failure=failure, status=status
        )
    if code == IndexErrorCode.PUBLIC_SEARCH_AUTHENTICATED:
        return PublicSearchAuthenticatedError(
            PUBLIC_SEARCH_AUTHENTICATED_MESSAGE, failure=failure, status=status
        )
    if code == IndexErrorCode.PUBLIC_SEARCH_DISABLED:
        return PublicSearchDisabledError(
            "Public Index search is disabled on this backend", failure=failure, status=status
        )
    if code == IndexErrorCode.SEARCH_NOT_FOUND:
        return PublicSearchNotFoundError(
            "Public Index search not found (unknown search id or wrong token)",
            failure=failure,
            status=status,
        )
    return PublicSearchError(
        f"Public Index search failed ({code or f'http_{status}'})", failure=failure, status=status
    )


# Wire adapter ----------------------------------------------------------------


def _require_str(source: Mapping[str, Any], key: str, context: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise PublicSearchError(f"{context}: missing {key}")
    return value


def _parse_item(payload: object) -> PublicSearchCitation:
    if not isinstance(payload, Mapping):
        raise PublicSearchError("public search citation was not an object")
    fields: dict[str, Any] = {str(key): value for key, value in payload.items()}
    return PublicSearchCitation(
        contribution_id=_require_str(fields, "contribution_id", "public search citation"),
        revision_id=_require_str(fields, "revision_id", "public search citation"),
        raw=fields,
    )


def _release_id(headers: Mapping[str, str] | None, payload: Mapping[str, Any]) -> str | None:
    """Header first (the Monitor's decision id travels outside the reviewed body), body second."""
    if headers is not None:
        from_header = headers.get(PUBLIC_SEARCH_RELEASE_HEADER)
        if isinstance(from_header, str) and from_header.strip():
            return from_header.strip()
    monitor = payload.get("monitor")
    release_id = monitor.get("release_id") if isinstance(monitor, Mapping) else None
    return release_id if isinstance(release_id, str) and release_id else None


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return value.strip() if isinstance(value, str) and value.strip() else None


def _search_id(headers: Mapping[str, str] | None, payload: Mapping[str, Any]) -> str:
    """``X-Index-Search-Id`` first (backend #1660); body ``search_id`` for older backends."""
    from_header = _header(headers, PUBLIC_SEARCH_ID_HEADER)
    if from_header is not None:
        return from_header
    return _require_str(payload, "search_id", "public search response")


def _charge_cents(headers: Mapping[str, str] | None, payload: Mapping[str, Any]) -> int:
    """Customer charge: header, then body ``amount_cents``, then legacy ``usage``."""
    from_header = _header(headers, PUBLIC_SEARCH_CHARGE_HEADER)
    if from_header is not None:
        if not from_header.isdigit():
            raise PublicSearchError("public search response: invalid customer charge header")
        return int(from_header)
    usage = payload.get("usage")
    charge = (
        usage.get("customer_charge_cents")
        if isinstance(usage, Mapping)
        else payload.get("amount_cents")
    )
    if isinstance(charge, bool) or not isinstance(charge, int):
        raise PublicSearchError("public search response: missing customer charge")
    return charge


def _parse_result(
    payload: object, mode: SearchMode, headers: Mapping[str, str] | None = None
) -> PublicSearchResult:
    if not isinstance(payload, Mapping):
        raise PublicSearchError("public search response was not an object")
    charge = _charge_cents(headers, payload)
    items = payload.get("citations")
    if not isinstance(items, list | tuple):
        raise PublicSearchError("public search response: missing citations list")
    partial_reason = payload.get("partial_reason")
    effective = payload.get("mode")
    return PublicSearchResult(
        search_id=_search_id(headers, payload),
        mode=SearchMode(effective) if isinstance(effective, str) else mode,
        status=_require_str(payload, "status", "public search response"),
        citations=tuple(_parse_item(item) for item in items),
        customer_charge_cents=charge,
        monitor_release_id=_release_id(headers, payload),
        response=_require_str(payload, "response", "public search response"),
        partial_reason=partial_reason if isinstance(partial_reason, str) else None,
        raw={key: value for key, value in payload.items() if key != "search_token"},
    )


def _is_delivery(payload: object) -> bool:
    """A 200 is a result only with the delivery envelope, never a lifecycle body.

    Backend ``_status_body`` answers a terminal Deep ``failed``/``cancelled`` with
    HTTP 200 and ``PublicSearchAccepted`` (``state``, ``result_available``).
    """
    return (
        isinstance(payload, Mapping)
        and "state" not in payload
        and "result_available" not in payload
        and ("usage" in payload or "citations" in payload or "response" in payload)
    )


def _lifecycle_state(payload: object) -> str:
    if isinstance(payload, Mapping):
        for key in ("state", "status"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return "pending"


def _raise_if_terminal(search_id: str, payload: object) -> str:
    """Return the lifecycle state, raising the typed error for a terminal non-delivery."""
    state = _lifecycle_state(payload)
    lowered = state.lower()
    if lowered not in _TERMINAL_NON_DELIVERY:
        return state
    failure = payload.get("failure") if isinstance(payload, Mapping) else None
    code = failure.get("code") if isinstance(failure, Mapping) else None
    retryable = failure.get("retryable") if isinstance(failure, Mapping) else None
    kind = (
        PublicSearchCancelledError
        if lowered in {"cancelled", "canceled"}
        else PublicSearchFailedError
    )
    raise kind(
        search_id,
        state,
        failure_code=code if isinstance(code, str) else None,
        retryable=retryable if isinstance(retryable, bool) else None,
    )


def _poll_after_seconds(headers: httpx.Headers, payload: object) -> float:
    candidates: list[object] = [headers.get("retry-after")]
    if isinstance(payload, Mapping):
        candidates.extend(payload.get(key) for key in ("poll_after_seconds", "retry_after_seconds"))
    for value in candidates:
        if value is None or isinstance(value, bool):
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(seconds):
            return min(MAX_PUBLIC_POLL_SECONDS, max(MIN_PUBLIC_POLL_SECONDS, seconds))
    return DEFAULT_PUBLIC_POLL_SECONDS


class PublicSearchClient:
    """Public search operations over one owned or borrowed sync transport.

    Bypasses ``request_json`` only to see the status code and ``Retry-After`` of
    a 202; error and exception handling stay the transport's own.
    """

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def _exchange(
        self,
        method: str,
        path: str,
        *,
        operation_id: str,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, httpx.Headers, Any]:
        transport = self._transport
        try:
            try:
                response = transport.client.request(
                    method,
                    path,
                    json=None if json_body is None else dict(json_body),
                    headers=None if headers is None else dict(headers),
                    timeout=transport.timeout_seconds if timeout_s is None else timeout_s,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                transport.exception_handler(method, path, exc, operation_id)
            if response.is_error:
                transport.error_handler(response, operation_id)
        except SynthError as error:
            translated = translate_public_search_error(error)
            if translated is error:
                raise
            raise translated from error
        payload: Any = {}
        if response.content:
            try:
                payload = response.json()
            except ValueError as exc:
                transport.decode_error_handler(method, path, response, exc, operation_id)
        return response.status_code, response.headers, payload

    def start(
        self,
        query: str,
        *,
        mode: SearchMode | str = SearchMode.FAST,
        max_results: int | None = None,
        idempotency_key: str | None = None,
        timeout_s: float | None = None,
    ) -> PublicSearchResult | PublicSearchHandle:
        """Send one public search. Fast returns a result; Deep returns a handle."""
        mode = SearchMode(mode)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        body: dict[str, Any] = {"mode": mode.value, "query": query}
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
                raise ValueError("max_results must be a positive integer")
            body["max_results"] = max_results
        if idempotency_key is not None:
            body["idempotency_key"] = idempotency_key
        status, headers, payload = self._exchange(
            "POST",
            _SEARCH_PATH,
            operation_id="index.public.search",
            json_body=body,
            timeout_s=timeout_s,
        )
        if status == 202 or not _is_delivery(payload):
            # Deep admission: 202, or 200 when the search was already terminal.
            if not isinstance(payload, Mapping):
                raise PublicSearchError("public search admission body was not an object")
            search_id = _require_str(payload, "search_id", "public search admission")
            _raise_if_terminal(search_id, payload)
            return PublicSearchHandle(
                self,
                search_id=search_id,
                token=_require_str(payload, "search_token", "public search admission"),
                mode=mode,
                poll_after_s=_poll_after_seconds(headers, payload),
            )
        return _parse_result(payload, mode, headers)

    def search(
        self,
        query: str,
        *,
        mode: SearchMode | str = SearchMode.FAST,
        max_results: int | None = None,
        idempotency_key: str | None = None,
        wait: bool = True,
        timeout_s: float = DEFAULT_PUBLIC_SEARCH_WAIT_SECONDS,
    ) -> PublicSearchResult | PublicSearchHandle:
        """Fast: the result. Deep: the result after polling, or the handle if ``wait=False``."""
        started = self.start(
            query, mode=mode, max_results=max_results, idempotency_key=idempotency_key
        )
        if isinstance(started, PublicSearchHandle) and wait:
            return started.wait(timeout_s=timeout_s)
        return started

    def handle(
        self, search_id: str, token: str, *, mode: SearchMode | str = SearchMode.DEEP
    ) -> PublicSearchHandle:
        """Reconnect to an admitted Deep search from a retained id and token.

        Deep only. A Fast result is final when ``start``/``search`` returns it and is
        not re-readable: the search token authorizes only the Deep lifecycle reads
        (status, result, cancel), so ``mode=FAST`` raises ``ValueError``.
        """
        resolved = SearchMode(mode)
        if resolved is SearchMode.FAST:
            raise ValueError(
                "Fast public search results are final and cannot be re-read; "
                "only an admitted Deep search has a handle"
            )
        return PublicSearchHandle(
            self,
            search_id=search_id,
            token=token,
            mode=resolved,
            poll_after_s=DEFAULT_PUBLIC_POLL_SECONDS,
        )

    def _read(
        self, search_id: str, token: str, *, timeout_s: float | None
    ) -> tuple[int, httpx.Headers, Any]:
        return self._exchange(
            "GET",
            _SEARCH_ITEM_PATH.format(search_id=search_id),
            operation_id="index.public.searches.get",
            headers={PUBLIC_SEARCH_TOKEN_HEADER: token},
            timeout_s=PUBLIC_POLL_REQUEST_TIMEOUT_SECONDS if timeout_s is None else timeout_s,
        )

    def _cancel(self, search_id: str, token: str) -> tuple[int, httpx.Headers, Any]:
        return self._exchange(
            "POST",
            _SEARCH_CANCEL_PATH.format(search_id=search_id),
            operation_id="index.public.searches.cancel",
            headers={PUBLIC_SEARCH_TOKEN_HEADER: token},
        )


_TERMINAL_NON_DELIVERY = frozenset({"failed", "cancelled", "canceled", "expired"})


class PublicSearchHandle:
    """An admitted Deep public search. Holds the token; never exposes it.

    ``repr``/``str`` omit the token. Persist ``search_id`` freely; persist the
    token only where you would persist a credential.
    """

    __slots__ = ("_client", "_result", "_token", "mode", "poll_after_s", "search_id")

    def __init__(
        self,
        client: PublicSearchClient,
        *,
        search_id: str,
        token: str,
        mode: SearchMode,
        poll_after_s: float,
    ) -> None:
        self._client = client
        self._token = token
        self._result: PublicSearchResult | None = None
        self.search_id = search_id
        self.mode = mode
        self.poll_after_s = poll_after_s

    def __repr__(self) -> str:
        return f"PublicSearchHandle(search_id={self.search_id!r}, mode={self.mode.value!r})"

    __str__ = __repr__

    @property
    def result(self) -> PublicSearchResult | None:
        """The delivered result once a poll has seen it; never blocks."""
        return self._result

    def poll(self, *, timeout_s: float | None = None) -> PublicSearchResult | PublicSearchStatus:
        """One status read. Updates ``poll_after_s`` from the backend's cadence."""
        status, headers, payload = self._client._read(
            self.search_id, self._token, timeout_s=timeout_s
        )
        if status == 200 and _is_delivery(payload):
            self._result = _parse_result(payload, self.mode, headers)
            return self._result
        state = _raise_if_terminal(self.search_id, payload)
        self.poll_after_s = _poll_after_seconds(headers, payload)
        return PublicSearchStatus(
            search_id=self.search_id, state=state, poll_after_s=self.poll_after_s
        )

    def wait(self, *, timeout_s: float = DEFAULT_PUBLIC_SEARCH_WAIT_SECONDS) -> PublicSearchResult:
        """Poll at the backend's cadence until delivered or the local deadline passes."""
        if not timeout_s > 0:
            raise ValueError("timeout_s must be positive")
        if self._result is not None:
            return self._result
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PublicSearchWaitTimeoutError(self)
            time.sleep(min(self.poll_after_s, remaining))
            outcome = self.poll(
                timeout_s=min(PUBLIC_POLL_REQUEST_TIMEOUT_SECONDS, max(1.0, remaining))
            )
            if isinstance(outcome, PublicSearchResult):
                return outcome

    def replay(self) -> PublicSearchResult:
        """Re-read the delivered result (cached after the first delivery)."""
        if self._result is not None:
            return self._result
        outcome = self.poll()
        if isinstance(outcome, PublicSearchResult):
            return outcome
        raise PublicSearchNotReadyError(
            f"Public search {self.search_id} is not delivered yet (state={outcome.state})"
        )

    def cancel(self) -> PublicSearchStatus:
        """Ask the backend to stop the search; completion may race the request."""
        _status, headers, payload = self._client._cancel(self.search_id, self._token)
        state = "cancellation_requested"
        if isinstance(payload, Mapping) and isinstance(payload.get("state"), str):
            state = payload["state"]
        return PublicSearchStatus(
            search_id=self.search_id,
            state=state,
            poll_after_s=_poll_after_seconds(headers, payload),
        )


# Copy for UIs and tool descriptions ----------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicSearchCopy:
    """Exact user-facing strings derived from backend capabilities, never hardcoded."""

    available: bool
    price: str
    limits: str
    privacy: str

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(line for line in (self.price, self.limits, self.privacy) if line)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "price": self.price,
            "limits": self.limits,
            "privacy": self.privacy,
        }


def _cents(value: int) -> str:
    if value == 0:
        return "free"
    return "1 cent" if value == 1 else f"{value} cents"


def public_search_copy(capability: PublicSearchCapability | None) -> PublicSearchCopy:
    """Render the backend's public search terms as sentences for a UI or tool text."""
    if capability is None:
        return PublicSearchCopy(
            available=False,
            price="Public search is not offered by this backend.",
            limits="",
            privacy="",
        )
    if not capability.enabled:
        return PublicSearchCopy(
            available=False,
            price="Public search is currently disabled on this backend.",
            limits="",
            privacy=capability.privacy_copy.strip(),
        )
    modes = capability.modes or tuple(capability.price_cents)
    price_parts: list[str] = []
    limit_parts: list[str] = []
    for mode in modes:
        name = mode.value.capitalize()
        cents = capability.price_cents.get(mode)
        if cents is None:
            price_parts.append(f"{name} search: price not published.")
        else:
            price_parts.append(f"{name} search is {_cents(cents)}.")
        limits = capability.limits.get(mode)
        if limits is not None:
            limit_parts.append(
                f"{name}: {limits.peer_per_minute} per minute and {limits.peer_per_day} per day "
                f"per caller; {limits.global_per_minute} per minute and "
                f"{limits.global_per_day} per day "
                "platform-wide."
            )
    query_retention = (
        "Public query content is not retained in durable storage."
        if capability.retention.public_query_days == 0
        else f"Public queries are retained for {capability.retention.public_query_days} days."
    )
    retention = (
        f"{query_retention} Processing state expires after "
        f"{capability.retention.private_processing_minutes} minutes."
    )
    privacy = f"{capability.privacy_copy.strip()} {retention}".strip()
    return PublicSearchCopy(
        available=True,
        price=" ".join(price_parts),
        limits=" ".join(limit_parts),
        privacy=privacy,
    )


class PublicSearchOperations:
    """Mixin giving a sync Index root the public search surface.

    Requires ``self._transport`` (``HttpTransport``) and ``self.capabilities()``.
    """

    _transport: HttpTransport

    def capabilities(self) -> Any:  # pragma: no cover - provided by the concrete root
        raise NotImplementedError

    def public_search(
        self,
        query: str,
        *,
        mode: SearchMode | str = SearchMode.FAST,
        max_results: int | None = None,
        idempotency_key: str | None = None,
        wait: bool = True,
        timeout_s: float = DEFAULT_PUBLIC_SEARCH_WAIT_SECONDS,
    ) -> PublicSearchResult | PublicSearchHandle:
        """Free public search; anonymous-only (see :class:`PublicSearchClient`).

        The backend refuses any credentialed request to the public route, so a
        keyed client gets :class:`PublicSearchAuthenticatedError` (409). Keyed
        callers should call ``search(...)`` instead, which is the paid keyed route.
        """
        return PublicSearchClient(self._transport).search(
            query,
            mode=mode,
            max_results=max_results,
            idempotency_key=idempotency_key,
            wait=wait,
            timeout_s=timeout_s,
        )

    def public_search_handle(
        self, search_id: str, token: str, *, mode: SearchMode | str = SearchMode.DEEP
    ) -> PublicSearchHandle:
        """Reconnect to an admitted Deep public search (Fast results are not re-readable)."""
        return PublicSearchClient(self._transport).handle(search_id, token, mode=mode)

    def public_search_capability(self) -> PublicSearchCapability | None:
        """The backend's public search block, or None on a backend without v0.2."""
        return self.capabilities().public_search

    def public_search_terms(self) -> PublicSearchCopy:
        """Price, limits and privacy copy exactly as the backend publishes them."""
        return public_search_copy(self.public_search_capability())
