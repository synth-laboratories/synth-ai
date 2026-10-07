"""Valid MCP, retry-identity and pagination controls paired with laws."""

import pytest
from boundary_probe import invoke_tool
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationId, OperationMetadata
from synth_ai.core.http.retry import idempotency_key_from_request
from synth_ai.mcp.research.registry import ToolDefinition, tool_schema
from synth_ai.sdk.pagination import page_from_wire


@pytest.mark.parametrize("arguments", [{"project_id": "project"}])
@pytest.mark.parametrize("entrypoint", ["registry", "server"])
def test_valid_closed_tool_arguments_reach_handler_once(arguments, entrypoint):
    calls = []

    def handler(arguments):
        calls.append(arguments)
        return {"project_id": arguments["project_id"]}

    tool = ToolDefinition(
        "research_probe",
        "offline",
        tool_schema({"project_id": {"type": "string"}}, required=["project_id"]),
        handler,
        required_scopes=("smr:read",),
    )
    assert invoke_tool(tool, arguments, entrypoint) == arguments
    assert calls == [arguments]


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"Idempotency-Key": "same", "idempotency-key": "same"}, {}),
        ({"Idempotency-Key": "same"}, {"idempotency_key": "same"}),
        ({}, {"idempotency_key": "same", "idempotency_key_run_create": "same"}),
    ],
)
def test_equal_idempotency_identities_remain_valid(headers, body):
    operation = OperationMetadata(
        OperationId("offline_write"), HttpMethod.POST, "/probe", mutation=True, idempotent=True
    )
    assert (
        idempotency_key_from_request(HttpRequest(operation, "/probe", headers=headers, body=body))
        == "same"
    )


@pytest.mark.parametrize(
    "payload,expected",
    [
        ([], ([], None, False)),
        ({"items": ["item"], "next_cursor": "next", "has_more": True}, (["item"], "next", True)),
        ({"data": ["item"], "next_cursor": None, "has_more": False}, (["item"], None, False)),
    ],
)
def test_valid_pages_preserve_data_and_continuation(payload, expected):
    assert page_from_wire(payload) == expected
