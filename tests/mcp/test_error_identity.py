"""MCP keeps specific server codes alongside coarse typed categories."""

import pytest

pytest.importorskip("mcp.types", reason="requires the synth-ai[mcp] extra")

from mcp.types import CallToolRequest, CallToolRequestParams
from synth_ai.errors import OutcomeUncertainError, ResultNotReadyError
from synth_ai.mcp.server import create_server, error_code

pytestmark = [pytest.mark.mcp, pytest.mark.asyncio]


async def test_tool_failure_passes_server_code_and_safe_diagnostic_fields():
    class Executor:
        async def available_tools(self):
            return frozenset({"index_search_list"})

        async def invoke(self, tool, arguments):
            raise ResultNotReadyError(
                "Wait for the result",
                code="result_not_ready_foo",
                status=409,
                request_id="server-request",
                operation="index.searches.list",
                retryable=True,
            )

    server = create_server(Executor())
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="index_search_list",
            arguments={"limit": 1},
        ),
    )
    result = (await server.request_handlers[CallToolRequest](request)).root
    error = result.structuredContent["error"]
    assert result.isError
    assert error["code"] == "result_not_ready_foo" and error["category"] == "result_not_ready"
    assert error["status"] == 409 and error["retryable"] is True
    assert error["request_id"] == "server-request" and error["operation"] == "index.searches.list"
    assert "Wait for the result" in error["message"] and "result_not_ready_foo" in error["message"]


async def test_invalid_server_code_falls_back_and_uncertainty_remains_custody_signal():
    assert error_code(ResultNotReadyError("waiting", code="untrusted code")) == "result_not_ready"
    assert (
        error_code(OutcomeUncertainError("unknown", code="activation_required", status=403))
        == "outcome_uncertain"
    )
