"""Closed arguments and opaque pagination laws, EX-14–EX-20.

See testing/specifications/sdk/core_research_migration.md,
October 5 stable boundary qualification; tests remain offline and tests-only.
"""

import pytest
from boundary_probe import invoke_tool
from synth_ai.core.contracts.pagination import extract_next_cursor
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationId, OperationMetadata
from synth_ai.core.http.retry import idempotency_key_from_request
from synth_ai.mcp.research.registry import ToolDefinition, tool_schema
from synth_ai.sdk.pagination import page_from_wire
from synth_ai.sdk.research.projects import _projects_page
from synth_ai.sdk.research.swarms import _swarms_page


@pytest.mark.parametrize("arguments", [{}, {"project_id": "project", "unknown": True}])
@pytest.mark.parametrize("entrypoint", ["registry", "server"])
def test_closed_mcp_arguments_refuse_before_handler__EX14(arguments, entrypoint):
    effects = []

    def handler(arguments):
        effects.append(arguments)
        return {"called": True}

    tool = ToolDefinition(
        "research_probe",
        "offline",
        tool_schema({"project_id": {"type": "string"}}, required=["project_id"]),
        handler,
        required_scopes=("smr:read",),
    )
    try:
        invoke_tool(tool, arguments, entrypoint)
    except (ValueError, TypeError):
        assert effects == [], "EX-14: argument validation occurred after handler"
        return
    pytest.fail("EX-14: missing/unknown closed-schema arguments reached handler")


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"Idempotency-Key": "A", "idempotency-key": "B"}, {}),
        ({"Idempotency-Key": "A"}, {"idempotency_key": "B"}),
        ({}, {"idempotency_key": "A", "idempotency_key_run_create": "B"}),
    ],
)
def test_ambiguous_idempotency_identity_refuses__EX15(headers, body):
    operation = OperationMetadata(
        OperationId("offline_write"), HttpMethod.POST, "/probe", mutation=True, idempotent=True
    )
    request = HttpRequest(operation, "/probe", body=body, headers=headers)
    try:
        idempotency_key_from_request(request)
    except (ValueError, TypeError):
        return
    pytest.fail("EX-15: contradictory header/body idempotency identities silently selected")


@pytest.mark.parametrize("payload", [{}, {"items": {}}, {"items": None}, {"data": "not-an-array"}])
def test_invalid_items_do_not_become_empty_page__EX16(payload):
    try:
        page_from_wire(payload)
    except (ValueError, TypeError):
        return
    pytest.fail("EX-16: invalid/missing page items silently become empty successful result")


def project_wire():
    return {
        "project_id": "project",
        "org_id": "org",
        "name": "project",
        "timezone": "UTC",
        "created_at": "2026-10-06T00:00:00Z",
        "updated_at": "2026-10-06T00:00:00Z",
    }


def swarm_wire():
    return {
        "run_id": "run",
        "project_id": "project",
        "org_id": "org",
        "public_state": "queued",
        "runbook": "lite",
        "trigger": "manual",
        "created_at": "2026-10-06T00:00:00Z",
        "updated_at": "2026-10-06T00:00:00Z",
    }


def test_terminal_null_is_not_previous_cursor__EX17():
    items, cursor, more = page_from_wire(
        {"items": [], "next_cursor": None, "cursor": "previous", "has_more": False}
    )
    assert cursor is None, "EX-17: terminal next_cursor null overwritten by prior cursor"
    assert items == [] and more is False


@pytest.mark.parametrize("kind", ["projects", "swarms"])
def test_full_terminal_page_does_not_invent_continuation__EX17(kind):
    wire = project_wire() if kind == "projects" else swarm_wire()
    parse = _projects_page if kind == "projects" else _swarms_page
    page = parse({"items": [wire], "next_cursor": None, "has_more": False}, limit=1)
    assert len(page.items) == 1
    assert page.next_cursor is None, "EX-17: explicit terminal page invents cursor from last item"
    assert page.has_more is False, "EX-17: explicit terminal page invents another page"


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_has_more_is_strict_boolean__EX18(value):
    try:
        page_from_wire({"items": [], "has_more": value})
    except (ValueError, TypeError):
        return
    pytest.fail("EX-18: non-boolean has_more silently coerced into continuation state")


@pytest.mark.parametrize(
    "parse", [page_from_wire, extract_next_cursor], ids=["sdk-page", "core-cursor"]
)
def test_opaque_cursor_is_preserved_exactly__EX19(parse):
    cursor = "  opaque/token==  "
    result = parse({"items": [], "next_cursor": cursor})
    actual = result[1] if isinstance(result, tuple) else result
    assert actual == cursor, "EX-19: opaque server cursor normalized by stripping whitespace"


@pytest.mark.parametrize("cursor", [123, True, {"value": "cursor"}])
def test_nonstring_cursor_is_not_fabricated__EX19(cursor):
    try:
        page_from_wire({"items": [], "next_cursor": cursor})
    except (ValueError, TypeError):
        return
    pytest.fail("EX-19: non-string cursor fabricated by str() coercion")


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [], "has_more": True, "next_cursor": None},
        {"items": [], "has_more": False, "next_cursor": "next"},
    ],
)
def test_contradictory_continuation_refuses__EX20(payload):
    try:
        page_from_wire(payload)
    except (ValueError, TypeError):
        return
    pytest.fail("EX-20: contradictory has_more/cursor declaration accepted")
