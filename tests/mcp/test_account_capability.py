"""Account MCP qualification stays separate from the existing capability receipt."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pydantic
import pytest
from synth_ai.contracts.fingerprint import QUALIFICATION_PYDANTIC_VERSION
from synth_ai.errors import (
    AuthenticationError,
    ContractError,
    PermissionDeniedError,
    UnsupportedOperationError,
)
from synth_ai.mcp.account import AccountUsageSnapshot, account_usage_snapshot
from synth_ai.mcp.account_capability import (
    ACCOUNT_MCP_SCHEMA_SHA256,
    AccountMcpCapability,
    account_mcp_schema_sha256_live,
    require_account_mcp,
)
from synth_ai.types.account_cloud import CurrentPlan
from synth_ai.types.capabilities import Capabilities
from tests.http_fixtures import capabilities_body


def release():
    return Capabilities.model_validate_json(json.dumps(capabilities_body()))


def account(**changes):
    now = datetime.now(UTC)
    return AccountMcpCapability(
        schema_sha256=ACCOUNT_MCP_SCHEMA_SHA256,
        evidence_sha256="b" * 64,
        release_manifest_sha256="a" * 64,
        environment="development",
        admission="qualified",
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=1),
        **changes,
    )


def test_retained_account_schema_fingerprint_matches_the_qualification():
    path = Path(__file__).resolve().parents[2] / "contracts/account-mcp-v1.json"
    assert (
        hashlib.sha256(path.read_text().strip().encode()).hexdigest() == ACCOUNT_MCP_SCHEMA_SHA256
    )


@pytest.mark.skipif(
    pydantic.__version__ != QUALIFICATION_PYDANTIC_VERSION,
    reason="Schema generation requires the qualified pydantic serializer",
)
def test_account_schema_fingerprint_is_the_qualified_export():
    assert account_mcp_schema_sha256_live() == ACCOUNT_MCP_SCHEMA_SHA256


def test_separate_capability_binds_release_and_window():
    selected = account()
    assert require_account_mcp(selected, release()) == selected
    with pytest.raises(UnsupportedOperationError):
        require_account_mcp(None, release())
    with pytest.raises(UnsupportedOperationError):
        require_account_mcp(
            selected.model_copy(update={"release_manifest_sha256": "f" * 64}), release()
        )
    with pytest.raises(UnsupportedOperationError):
        require_account_mcp(
            selected.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}),
            release(),
        )
    with pytest.raises(ContractError):
        require_account_mcp(selected.model_copy(update={"schema_sha256": "f" * 64}), release())


def test_unqualified_candidate_requires_an_explicit_nonproduction_pin():
    selected = account().model_copy(update={"admission": "candidate", "evidence_sha256": None})
    base = release()
    candidate = base.model_copy(update={"admission": "candidate"})
    with pytest.raises(UnsupportedOperationError):
        require_account_mcp(selected, candidate)
    assert require_account_mcp(selected, candidate, candidate_manifest_sha256="a" * 64) == selected
    with pytest.raises(ContractError):
        require_account_mcp(selected, candidate, candidate_manifest_sha256="f" * 64)
    with pytest.raises(UnsupportedOperationError):
        require_account_mcp(selected.model_copy(update={"admission": "qualified"}), base)


def test_snapshot_requires_a_typed_value_for_ready_state():
    value = {
        "schema": "synth.account-usage.v1",
        "plan": {"state": "unavailable"},
        "balance": {"state": "unavailable"},
        "usage": {"state": "unavailable"},
    }
    parsed = AccountUsageSnapshot.model_validate_json(json.dumps(value))
    assert json.loads(parsed.model_dump_json())["schema"] == "synth.account-usage.v1"
    value["usage"] = {"state": "ready"}
    with pytest.raises(ValueError):
        AccountUsageSnapshot.model_validate_json(json.dumps(value))


def test_snapshot_preserves_partial_observations_without_error_details():
    class Account:
        async def plan(self, **kwargs):
            return CurrentPlan(state="unavailable", observed_at=datetime.now(UTC))

        async def balance(self, **kwargs):
            raise PermissionDeniedError("secret detail", code="account_scope_required")

        async def usage(self, *args, **kwargs):
            raise PermissionDeniedError("private", code="other")

    client = type("Client", (), {"account": Account()})()
    value = asyncio.run(account_usage_snapshot(client, None)).model_dump(mode="json", by_alias=True)
    assert value["plan"]["state"] == "ready"
    assert value["balance"]["state"] == "unavailable"
    assert value["balance"]["code"] == "account_scope_required"
    assert value["usage"]["state"] == "unavailable"
    assert "secret detail" not in str(value) and "private" not in str(value)


def test_revoked_credential_is_not_downgraded_to_partial_success():
    class Account:
        async def plan(self, **kwargs):
            raise AuthenticationError("credential revoked")

        async def balance(self, **kwargs):
            raise PermissionDeniedError("unavailable")

        async def usage(self, *args, **kwargs):
            raise PermissionDeniedError("unavailable")

    client = type("Client", (), {"account": Account()})()
    with pytest.raises(AuthenticationError):
        asyncio.run(account_usage_snapshot(client, None))
