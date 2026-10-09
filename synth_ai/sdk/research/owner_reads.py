"""Scope-bound reads from an owner API with an independently issued credential.

# See: testing/specifications/sdk/owner_reads.md
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

from synth_ai.core.contracts.json_value import JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.streaming import SseEvent
from synth_ai.core.http.transport import HttpTransport

Owner = Literal["orchestra", "sublinear"]
_PATH = re.compile(r"/[A-Za-z0-9_./-]*\Z")
_SCOPE_KEYS = frozenset({"organization_id", "project_id", "run_id"})


def _canonical_id(value: str) -> None:
    parsed = UUID(value)
    if str(parsed) != value or parsed.int == 0:
        raise ValueError("owner read identities require canonical nonzero UUIDs")


def _path(value: str) -> str:
    if not _PATH.fullmatch(value) or "//" in value:
        raise ValueError("owner read path must be an unencoded absolute path")
    if any(part in {".", ".."} for part in value.split("/")):
        raise ValueError("owner read path cannot traverse directories")
    return value


def _origin(value: str) -> str:
    if any(ord(char) <= 32 or ord(char) >= 127 for char in value) or "\\" in value:
        raise ValueError("owner origin must be an unambiguous ASCII URL")
    parsed = urlsplit(value)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("owner base URL must contain only a trusted origin")
    loopback = parsed.hostname == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("owner reads require HTTPS or explicit loopback HTTP")
    if parsed.port == 0:
        raise ValueError("owner origin cannot use port zero")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class OwnerReadScope:
    organization_id: str
    project_id: str
    run_id: str

    def __post_init__(self) -> None:
        for value in (self.organization_id, self.project_id, self.run_id):
            _canonical_id(value)


@dataclass(frozen=True, slots=True)
class OwnerReadAccess:
    """Owner-issued access, not a backend API key or a client authority check.

    The receiving owner must verify token audience, exact scope and current
    reader authority. These local fields prevent accidental credential mixing;
    they cannot replace server authentication or revocation checks.
    """

    owner: Owner
    origin: str
    scope: OwnerReadScope
    path_prefix: str
    token: str = field(repr=False)
    expires_at: datetime

    def __post_init__(self) -> None:
        if self.owner not in {"orchestra", "sublinear"}:
            raise ValueError("unsupported read owner")
        if _origin(self.origin) != self.origin:
            raise ValueError("owner read origin must be normalized")
        if _path(self.path_prefix) == "/" or self.path_prefix.endswith("/"):
            raise ValueError("owner reads require an explicit resource prefix")
        if not self.token or any(ord(char) <= 32 or ord(char) >= 127 for char in self.token):
            raise ValueError("owner read credential must be a nonempty header token")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("owner read expiry must include a timezone")

    @classmethod
    def from_response(
        cls, value: JsonValue, *, owner: Owner, scope: OwnerReadScope
    ) -> OwnerReadAccess:
        """Decode the issuer reply and refuse a changed owner or read scope."""
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema_version",
                "owner",
                "origin",
                "scope",
                "path_prefix",
                "token",
                "expires_at_ms",
            }
            or value.get("schema_version") != "synth.owner-read-access.v1"
            or value.get("owner") != owner
        ):
            raise ValueError("owner read issuer response invalid")
        returned_scope = value["scope"]
        if returned_scope != {
            "organization_id": scope.organization_id,
            "project_id": scope.project_id,
            "stream_id": scope.run_id,
        }:
            raise ValueError("owner read issuer scope mismatch")
        expiry = value["expires_at_ms"]
        if type(expiry) is not int or expiry <= 0:
            raise ValueError("owner read issuer expiry invalid")
        if not all(isinstance(value[key], str) for key in ("origin", "path_prefix", "token")):
            raise ValueError("owner read issuer credential invalid")
        expected_prefix = (
            f"/v1/streams/{scope.run_id}"
            if owner == "orchestra"
            else f"/v1/runs/{scope.run_id}/planning"
        )
        if value["path_prefix"] != expected_prefix:
            raise ValueError("owner read issuer resource mismatch")
        return cls(
            owner=owner,
            origin=value["origin"],
            scope=scope,
            path_prefix=value["path_prefix"],
            token=value["token"],
            expires_at=datetime.fromtimestamp(expiry / 1000, UTC),
        )

    def request(
        self, resource: str, query: Mapping[str, JsonValue] | None
    ) -> tuple[str, dict[str, JsonValue]]:
        if datetime.now(UTC) >= self.expires_at:
            raise ValueError("owner read access expired")
        if resource:
            _path(resource)
        supplied = dict(query or {})
        if _SCOPE_KEYS.intersection(supplied):
            raise ValueError("owner read scope cannot be overridden")
        supplied["organization_id"] = self.scope.organization_id
        supplied["project_id"] = self.scope.project_id
        return self.path_prefix + resource, supplied


class OwnerReadClient:
    """Read JSON, bytes and resumable SSE from one scoped owner resource.

    No backend transport or fallback is retained. Redirects refuse, and request
    callers cannot replace authorization headers or select another origin.
    Owner response bytes/cursors remain unmodified for the owner's decoder.
    """

    def __init__(
        self, *, owner: Owner, base_url: str, access: OwnerReadAccess, timeout_seconds: float = 30.0
    ) -> None:
        if owner != access.owner or _origin(base_url) != access.origin:
            raise ValueError("owner read credential audience mismatch")
        self.access = access
        self._transport = HttpTransport(
            base_url=_origin(base_url),
            headers={"Authorization": f"Bearer {access.token}"},
            timeout_seconds=timeout_seconds,
            follow_redirects=False,
        )

    def read_json(
        self, resource: str = "", *, query: Mapping[str, JsonValue] | None = None
    ) -> JsonValue:
        path, params = self.access.request(resource, query)
        return self._transport.request_json("GET", path, params=params)

    def read_bytes(
        self, resource: str = "", *, query: Mapping[str, JsonValue] | None = None
    ) -> bytes:
        path, params = self.access.request(resource, query)
        return self._transport.request_bytes("GET", path, params=params)

    def subscribe(
        self,
        resource: str,
        *,
        query: Mapping[str, JsonValue] | None = None,
        last_event_id: str | None = None,
    ) -> Iterator[SseEvent]:
        path, params = self.access.request(resource, query)
        yield from self._transport.stream_sse(path, params=params, last_event_id=last_event_id)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> OwnerReadClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AsyncOwnerReadClient:
    """Native async owner reader with the same scope and transport restrictions."""

    def __init__(
        self, *, owner: Owner, base_url: str, access: OwnerReadAccess, timeout_seconds: float = 30.0
    ) -> None:
        if owner != access.owner or _origin(base_url) != access.origin:
            raise ValueError("owner read credential audience mismatch")
        self.access = access
        self._transport = AsyncHttpTransport(
            base_url=_origin(base_url),
            headers={"Authorization": f"Bearer {access.token}"},
            timeout_seconds=timeout_seconds,
            follow_redirects=False,
        )

    async def read_json(
        self, resource: str = "", *, query: Mapping[str, JsonValue] | None = None
    ) -> JsonValue:
        path, params = self.access.request(resource, query)
        return await self._transport.request_json("GET", path, params=params)

    async def read_bytes(
        self, resource: str = "", *, query: Mapping[str, JsonValue] | None = None
    ) -> bytes:
        path, params = self.access.request(resource, query)
        return await self._transport.request_bytes("GET", path, params=params)

    async def subscribe(
        self,
        resource: str,
        *,
        query: Mapping[str, JsonValue] | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[SseEvent]:
        path, params = self.access.request(resource, query)
        async for event in self._transport.stream_sse(
            path, params=params, last_event_id=last_event_id
        ):
            yield event

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncOwnerReadClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


__all__ = ["AsyncOwnerReadClient", "OwnerReadAccess", "OwnerReadClient", "OwnerReadScope"]
