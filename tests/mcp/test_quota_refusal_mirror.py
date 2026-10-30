"""MCP tool failures mirror the typed quota fields (SYN-4844)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp.types", reason="requires the synth-ai[mcp] extra")

from synth_ai.core.http.quota import decode_quota_refusal
from synth_ai.errors import ContractError
from synth_ai.mcp.contracts import QuotaFacts, ToolFailure
from synth_ai.mcp.server import error_category, error_code, quota_facts

pytestmark = pytest.mark.mcp

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/smr_quota_refusals_v1.json").read_text()
)["cases"]


def _case(name: str) -> tuple[int, bytes]:
    case = FIXTURES[name]
    return case["status"], json.dumps(case["body"]).encode()


def test_mcp_failure_mirrors_the_typed_quota_fields():
    status, body = _case("plan_active_limit_403")
    error = decode_quota_refusal(status, body)
    failure = ToolFailure(
        code=error_code(error),
        category=error_category(error),
        message=str(error)[:512],
        status=error.status,
        retryable=error.retryable,
        quota=quota_facts(error),
    )
    data = json.loads(failure.model_dump_json())
    assert (data["code"], data["category"]) == (
        "smr_project_plan_active_limit_reached",
        "quota_exceeded",
    )
    assert data["quota"] == {
        "scope": "organization",
        "limit": 4,
        "used": 4,
        "remedy": "Archive an active project or upgrade the plan.",
    }


def test_mcp_failure_omits_quota_for_other_errors():
    failure = ToolFailure(code="conflict", message="x", quota=quota_facts(ContractError("x")))
    assert "quota" not in json.loads(failure.model_dump_json())
    assert QuotaFacts(scope="project", limit=None, used=0).limit is None
