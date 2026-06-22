"""Open Research test fixtures.

Avoids touching the real backend by stubbing ``httpx.Client`` with a
deterministic ``MockTransport``. Tests assert on request method/path/
headers/body so any drift from the locked HTTP contract is caught at
unit-test time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx


@dataclass
class RecordedRequest:
    method: str
    path: str
    params: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    json_body: dict | None = None
    raw_body: bytes | None = None


@dataclass
class ScriptedResponse:
    status_code: int = 200
    json_body: dict | list | None = None
    raw_body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str = "application/json"


def make_mock_transport(
    responses: Iterable[ScriptedResponse],
    recorder: list[RecordedRequest],
) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        params = {k: v for k, v in request.url.params.items()}
        body_bytes = request.content or b""
        parsed_body: dict | None
        if body_bytes:
            try:
                parsed_body = json.loads(body_bytes.decode("utf-8"))
                if not isinstance(parsed_body, dict):
                    parsed_body = None
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed_body = None
        else:
            parsed_body = None
        recorder.append(
            RecordedRequest(
                method=request.method,
                path=request.url.path,
                params=params,
                headers={k.lower(): v for k, v in request.headers.items()},
                json_body=parsed_body,
                raw_body=body_bytes or None,
            )
        )
        if not queue:
            raise AssertionError(
                f"Unexpected request {request.method} {request.url.path}: queue exhausted"
            )
        scripted = queue.pop(0)
        headers = dict(scripted.headers)
        if scripted.raw_body is not None:
            headers.setdefault("content-type", scripted.content_type)
            return httpx.Response(
                scripted.status_code,
                content=scripted.raw_body,
                headers=headers,
            )
        if scripted.json_body is not None:
            return httpx.Response(
                scripted.status_code,
                json=scripted.json_body,
                headers=headers,
            )
        return httpx.Response(scripted.status_code, headers=headers)

    return httpx.MockTransport(handler)


def patch_client_transport(
    open_research_client,
    responses: Iterable[ScriptedResponse],
    recorder: list[RecordedRequest],
) -> None:
    """Replace the httpx.Client on a built OpenResearchClient with a mock."""

    open_research_client._client.close()
    open_research_client._client = httpx.Client(
        base_url=open_research_client.backend_base,
        headers=open_research_client._client.headers,
        timeout=open_research_client.timeout_seconds,
        transport=make_mock_transport(responses, recorder),
    )
