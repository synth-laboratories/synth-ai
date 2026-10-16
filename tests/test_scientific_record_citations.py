"""R06: retained-record citation verification through SDK and MCP.

Backend route: GET /smr/projects/{project_id}/research/records/{record_id}/citations
(forge.citation-verification.v1). Transport is a recording fake; no network.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from synth_ai.mcp.research.registry import READ_SCOPES
from synth_ai.mcp.research.server import ResearchMcpServer
from synth_ai.mcp.research.tools.scientific_records import build_scientific_record_tools
from synth_ai.sdk.research.contracts.forge_citations import CitationVerification
from synth_ai.sdk.research.scientific_records import (
    AsyncScientificRecordsAPI,
    ScientificRecordsAPI,
)

RECORD = {
    "authority": "forge",
    "kind": "result",
    "record_id": "result",
    "revision": "1",
    "digest_sha256": hashlib.sha256(b"result").hexdigest(),
}
ARTIFACT = {
    "authority": "artifact-platform",
    "kind": "object",
    "record_id": "00000000-0000-0000-0000-0000000000d4",
    "revision": "3",
    "digest_sha256": hashlib.sha256(b"cited evidence bytes").hexdigest(),
}
KEPT = {**ARTIFACT, "record_id": "00000000-0000-0000-0000-0000000000e5"}


def verification(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "forge.citation-verification.v1",
        "record": RECORD,
        "citations": [
            {
                "reference": ARTIFACT,
                "status": "dangling",
                "code": "forbidden",
                "authority_code": "reference_unavailable",
            },
            {"reference": KEPT, "status": "retained", "code": "", "authority_code": ""},
        ],
        "retained": False,
    }
    document.update(overrides)
    return document


class _Transport:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        self.calls: list[tuple[str, str, Any]] = []

    def request_json(self, method: str, path: str, *, params=None, **_: Any):
        self.calls.append((method, path, params))
        return self.document


class _AsyncTransport(_Transport):
    async def request_json(self, method: str, path: str, *, params=None, **_: Any):
        self.calls.append((method, path, params))
        return self.document


PATH = "/smr/projects/project/research/records/result/citations"


def test_sync_verify_citations_reads_typed_states() -> None:
    transport = _Transport(verification())
    result = ScientificRecordsAPI(transport).verify_citations("project", "result", revision=1)
    assert isinstance(result, CitationVerification)
    assert transport.calls == [("GET", PATH, {"revision": 1})]
    assert [check.status for check in result.citations] == ["dangling", "retained"]
    assert result.citations[0].reference.model_dump() == ARTIFACT
    assert result.citations[0].authority_code == "reference_unavailable"
    assert result.retained is False


def test_async_verify_citations_has_parity() -> None:
    transport = _AsyncTransport(verification())
    result = asyncio.run(AsyncScientificRecordsAPI(transport).verify_citations("project", "result"))
    assert transport.calls == [("GET", PATH, None)]
    assert result.citations[0].status == "dangling"


@pytest.mark.parametrize(
    "overrides",
    [
        {"record": {**RECORD, "record_id": "other"}},
        {"record": {**RECORD, "revision": "2"}},
        {"record": {**RECORD, "authority": "artifact-platform"}},
        {"retained": True},
        {
            "citations": [{"reference": {**RECORD, "record_id": "inner"}, "status": "retained"}],
            "retained": True,
        },
    ],
)
def test_verify_citations_refuses_identity_drift(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ScientificRecordsAPI(_Transport(verification(**overrides))).verify_citations(
            "project", "result", revision=1
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "forge.citation-verification.v2"},
        {"citations": [{"reference": ARTIFACT, "status": "gone"}]},
    ],
)
def test_verify_citations_refuses_unknown_versions_and_states(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ScientificRecordsAPI(_Transport(verification(**overrides))).verify_citations(
            "project", "result"
        )


class _Client:
    def __init__(self, transport: _Transport) -> None:
        self.records = ScientificRecordsAPI(transport)

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


def test_mcp_tool_is_read_scoped_and_returns_wire_document() -> None:
    transport = _Transport(verification())
    tools = {
        tool.name: tool for tool in build_scientific_record_tools(lambda args: _Client(transport))
    }
    tool = tools["research_verify_citations"]
    assert tool.required_scopes == READ_SCOPES
    assert tool.required_scopes == tools["research_get_record"].required_scopes
    assert tool.input_schema == tools["research_get_record"].input_schema
    answer = tool.handler({"project_id": "project", "record_id": "result", "revision": 1})
    assert transport.calls == [("GET", PATH, {"revision": 1})]
    assert answer == CitationVerification.model_validate(verification()).model_dump(mode="json")
    with pytest.raises(ValueError):
        tool.handler({"project_id": "project", "record_id": "result", "revision": 0})


def test_mcp_tool_is_advertised_in_the_stable_set_next_to_get_record() -> None:
    names = {tool["name"] for tool in ResearchMcpServer().list_tool_payload()}
    assert {"research_get_record", "research_verify_citations"} <= names
