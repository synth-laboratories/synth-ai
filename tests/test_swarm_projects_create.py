"""`client.swarms.projects.create` is a typed, keyed, scoped mutation present in every surface (SYN-4509)."""

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from synth_ai.contracts import swarms as operations
from synth_ai.mcp import catalog
from synth_ai.mcp.contracts import ProjectCreate
from synth_ai.sdk.swarms.client import AsyncProjects, Projects
from synth_ai.sdk.swarms.types.projects import ProjectCreateRequest, ProjectTemplate

pytestmark = pytest.mark.unit

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def test_operation_is_a_keyed_scoped_mutation():
    op = operations.CREATE_PROJECT
    assert (op.id, op.method, op.path) == ("swarms.projects.create", "POST", "/swarms/projects")
    assert op.mutation and op.keyed and op.required_scope == "swarms:write" and not op.may_charge
    assert op in operations.ALL_OPERATIONS


def test_request_is_closed_and_bounded():
    ok = ProjectCreateRequest(name="n", template=ProjectTemplate.REPORT_NATIVE, objective="o")
    assert ok.template.value == "report_native"
    for bad in (
        {"name": "", "template": ProjectTemplate.REPORT_NATIVE, "objective": "o"},
        {"name": "n", "template": ProjectTemplate.REPORT_NATIVE, "objective": "  "},
        {"name": "n", "template": ProjectTemplate.REPORT_NATIVE, "objective": "x" * 8193},
        {"name": "n", "template": ProjectTemplate.REPORT_NATIVE, "objective": "o", "extra": 1},
    ):
        with pytest.raises(ValidationError):
            ProjectCreateRequest(**bad)
    with pytest.raises(ValidationError):
        ProjectCreateRequest(name="n", template="unknown", objective="o")


def test_sync_and_async_clients_expose_create():
    for cls in (Projects, AsyncProjects):
        parameters = inspect.signature(cls.create).parameters
        assert list(parameters)[:2] == ["self", "name"]
        assert {"objective", "template", "mutation"} <= set(parameters)
        assert parameters["objective"].kind is inspect.Parameter.KEYWORD_ONLY


def test_mcp_tool_is_registered_with_an_idempotency_key():
    tool = next(t for t in catalog.TOOLS if t.name == "swarms_project_create")
    assert tool.definition is operations.CREATE_PROJECT
    assert "idempotency_key" in ProjectCreate.model_fields
    assert tool.request is ProjectCreate


def test_shipped_contracts_include_the_operation_and_tool():
    swarms = json.loads((CONTRACTS / "swarms-v1.json").read_text())
    assert "swarms.projects.create" in {op["id"] for op in swarms["operations"]}
    assert "ProjectCreateRequest" in swarms["models"]
    mcp = json.loads((CONTRACTS / "mcp-v1.json").read_text())
    assert "swarms_project_create" in {tool["name"] for tool in mcp["tools"]}
