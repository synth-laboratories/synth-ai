"""The run-history tail position survives the SDK and MCP boundaries."""

import json

from synth_ai.mcp.catalog import TOOLS
from synth_ai.sdk.swarms.types.runs import AdmittedResourceGrant, SwarmEventPage


def test_tail_cursor_is_retained_and_older_pages_still_decode():
    page = {
        "metadata": {
            "contract": "swarms.v1", "request_id": "request-1",
            "operation_id": None, "observed_at": "2026-10-28T00:00:00Z",
        },
        "items": [], "complete": True, "next_cursor": None,
        "resume_cursor": "opaque-position",
    }
    parsed = SwarmEventPage.model_validate_json(json.dumps(page))
    assert parsed.resume_cursor == "opaque-position"
    spec = next(tool for tool in TOOLS if tool.name == "swarms_events")
    assert spec.validate_result(page)["resume_cursor"] == "opaque-position"
    del page["resume_cursor"]
    assert SwarmEventPage.model_validate_json(json.dumps(page)).resume_cursor is None


def test_admitted_grant_summary_keeps_resources_and_ignores_credential_fields():
    grant = AdmittedResourceGrant.model_validate_json(json.dumps({
        "grant_id": "grant-1", "plugin": "index.search", "version": 1,
        "roles": ["worker"], "limits": {"calls": 2}, "role_limits": {},
        "models": [], "collections": [], "selection": None, "funding_lane": "platform",
        "credential_refs": ["private-credential"],
    }))
    assert grant.limits == {"calls": 2}
    assert "private-credential" not in grant.model_dump_json()
