"""Separate account MCP qualification; the existing capabilities.v1 stays unchanged."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime

from synth_ai.core.capabilities import validate_release
from synth_ai.errors import ContractError, UnsupportedOperationError
from synth_ai.types.base import Digest, Model
from synth_ai.types.capabilities import Capabilities, TrustEnvironment

from .account import AccountUsageSnapshot

ACCOUNT_MCP_OPERATION = "account.snapshot"
ACCOUNT_MCP_SCOPES = ("account:billing:read", "account:usage:read")
# Generated with the SDK's qualified pydantic version (2.12.5). Runtime
# negotiation must not generate schemas concurrently or depend on local versions.
ACCOUNT_MCP_SCHEMA_SHA256 = "efe29c7f7c78d11529fb20a13cb6041b4dd812f6e17c94536df6e1c0e9aac0c7"


def export_account_mcp_schema() -> str:
    """Export the closed response and required-scope contract for release tests."""
    schema = {
        "contract": "account.v1",
        "operation": ACCOUNT_MCP_OPERATION,
        "scopes": ACCOUNT_MCP_SCOPES,
        "result": AccountUsageSnapshot.model_json_schema(),
    }
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def account_mcp_schema_sha256_live() -> str:
    return hashlib.sha256(export_account_mcp_schema().encode()).hexdigest()


class AccountMcpCapability(Model):
    schema_version: Literal["synth.account-mcp-capability.v1"] = "synth.account-mcp-capability.v1"
    contract: Literal["account.v1"] = "account.v1"
    operation: Literal["account.snapshot"] = ACCOUNT_MCP_OPERATION
    schema_sha256: Digest
    evidence_sha256: Digest | None
    release_manifest_sha256: Digest
    environment: TrustEnvironment
    admission: Literal["candidate", "qualified"]
    issued_at: AwareDatetime
    expires_at: AwareDatetime


def require_account_mcp(
    account: AccountMcpCapability | None,
    release: Capabilities,
    *,
    candidate_manifest_sha256: str | None = None,
    now: datetime | None = None,
) -> AccountMcpCapability:
    """Require separate account evidence and the same current release identity."""
    validate_release(release, None, candidate_manifest_sha256)
    instant = now or datetime.now(UTC)
    if (
        account is None
        or release.release is None
        or release.admission is None
        or account.release_manifest_sha256 != release.release.manifest_sha256
        or account.environment != release.release.environment
        or account.admission != release.admission
        or account.admission == "qualified"
        and account.evidence_sha256 is None
        or not account.issued_at <= instant < account.expires_at
        or account.admission == "candidate"
        and candidate_manifest_sha256 is None
        or account.admission == "candidate"
        and account.environment == "production"
    ):
        raise UnsupportedOperationError(
            "Account MCP qualification unavailable",
            code="account_capability_unavailable",
            status=503,
        )
    if account.schema_sha256 != ACCOUNT_MCP_SCHEMA_SHA256:
        raise ContractError("Account MCP schema mismatch", code="schema_mismatch")
    return account
