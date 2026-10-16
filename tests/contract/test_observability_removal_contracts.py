"""MX-01: unsupported diagnostics are absent at every public entry point."""

import pytest
from synth_ai.mcp.research.server import ResearchMcpServer
from synth_ai.sdk.research.session.client import ResearchSession
from synth_ai.sdk.research.session.runs import RunHandle, RunsAPI


@pytest.mark.parametrize(
    "name", ["list_run_participants", "get_run_artifact_progress", "list_run_actor_logs"]
)
def test_unsupported_session_method_removed(name):
    assert not hasattr(ResearchSession, name)


@pytest.mark.parametrize("owner", [RunHandle, RunsAPI])
@pytest.mark.parametrize("name", ["participants", "artifact_progress", "actor_logs"])
def test_unsupported_run_method_removed(owner, name):
    assert not hasattr(owner, name)


@pytest.mark.parametrize("advanced", [False, True])
def test_unsupported_tools_removed(advanced):
    server = ResearchMcpServer(
        api_key="offline-dummy",
        backend_base="http://offline.invalid",
        include_advanced_tools=advanced,
    )
    names = {tool.name for tool in server.tool_definitions()}
    assert not names.intersection(
        {
            "research_list_run_participants",
            "research_get_run_artifact_progress",
            "research_list_run_actor_logs",
        }
    )
