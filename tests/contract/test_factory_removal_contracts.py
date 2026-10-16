"""MX-02/MX-06: removed Factory calls have no reachable SDK consumers."""

import pytest
from synth_ai.sdk.research.advanced_factories import (
    ResearchFactoriesTagScopesAPI,
    ResearchFactoriesTagSessionsAPI,
)
from synth_ai.sdk.research.session.client import ResearchSession
from synth_ai.sdk.research.session.tag import TagAPI


@pytest.mark.parametrize(
    "owner,name",
    [
        (ResearchSession, "patch_factory_status_compat"),
        (ResearchSession, "get_tag_scope_factory_context"),
        (ResearchSession, "get_tag_session_factory_context"),
        (TagAPI, "get_factory_context"),
        (ResearchFactoriesTagSessionsAPI, "get_factory_context"),
        (ResearchFactoriesTagScopesAPI, "get_factory_context"),
    ],
)
def test_dead_factory_call_removed(owner, name):
    assert not hasattr(owner, name), f"MX-02/MX-06: {owner.__name__}.{name} remains reachable"


def test_swarm_types_do_not_depend_on_factory_types():
    from typing import get_type_hints

    from synth_ai.sdk.research.contracts.swarms import Swarm, SwarmSpec
    from synth_ai.sdk.research.session.runs import RunHandle
    from synth_ai.sdk.research.swarms import AsyncSwarmHandle, SwarmHandle

    for owner in (Swarm, SwarmSpec):
        assert get_type_hints(owner)["effort_id"] == str | None
    for owner in (SwarmHandle, AsyncSwarmHandle):
        assert not hasattr(owner, "trace_store")
        assert not hasattr(owner, "traces")  # Factory trace queries live in the optional module
    assert not hasattr(RunHandle, "results")  # Factory results stay on factories.results


def test_factory_mcp_tools_are_explicitly_optional():
    from synth_ai.mcp.research.server import ResearchMcpServer

    defaults = ResearchMcpServer(api_key="offline-dummy", backend_base="http://offline.invalid")
    optional = ResearchMcpServer(
        api_key="offline-dummy", backend_base="http://offline.invalid", include_advanced_tools=True
    )
    factory_tools = {
        name
        for name in optional.available_tool_names()
        if "factory" in name
        or "factories" in name
        or name in {"research_create_effort", "research_get_effort", "research_patch_effort"}
    }
    assert "research_create_factory" in factory_tools
    assert "research_create_effort" in factory_tools
    assert not factory_tools.intersection(defaults.available_tool_names())

    assert {"intern_effort_board", "intern_effort_detail"} <= set(defaults.available_tool_names())
