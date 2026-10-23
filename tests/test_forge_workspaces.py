"""Public workspace clients and MCP discovery against synthetic wire fixtures."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts.fingerprint import schema_sha256
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.forge.contracts.contracts import contract_digest
from synth_ai.sdk.forge.contracts.projects import (
    ProjectMember,
    ProjectOperation,
    ProjectPrincipal,
    ProjectReceipt,
    ProjectState,
)
from synth_ai.sdk.forge.skill import research_skill
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.http_fixtures import capabilities_body


def receipt_and_headers():
    body = capabilities_body()
    body["contracts"] = [{"contract": "core.v1", "schema_sha256": schema_sha256("core.v1")}]
    body["operations"] = [
        {
            "operation_id": name,
            "contract": "core.v1",
            "state": "enabled",
            "surfaces": ["http", "mcp"],
            "reason": None,
        }
        for name in ("forge.projects.apply", "forge.projects.get")
    ]
    snapshot = {k: v for k, v in body.items() if k not in {"metadata", "capability_sha256"}}
    body["capability_sha256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(json.dumps(snapshot))
    )
    headers = {
        "X-Synth-Contract": "core.v1",
        "X-Synth-Schema-SHA256": schema_sha256("core.v1"),
        "X-Synth-Capabilities-SHA256": body["capability_sha256"],
        "X-Synth-Release-Manifest-SHA256": body["release"]["manifest_sha256"],
    }
    return body, headers


def fixture_transport():
    capability, headers = receipt_and_headers()
    calls, retained = [], {}

    def respond(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        calls.append(request)
        operation = ProjectOperation.model_validate_json(request.content)
        assert request.headers["Idempotency-Key"] == operation.operation_id
        assert (
            request.headers["X-Synth-Intent-SHA256"] == hashlib.sha256(request.content).hexdigest()
        )
        assert operation.project_id in request.url.path
        principal = ProjectPrincipal(
            organization_id="org",
            subject_id="human",
            subject_kind="human",
            credential_id="credential",
            required_scope="forge:write",
        )
        if operation.operation_id not in retained:
            state = ProjectState(
                organization_id="org",
                project_id=operation.project_id,
                revision=1,
                title=operation.change.title,
                objective=operation.change.objective,
                members=(ProjectMember(subject_kind="human", subject_id="human", role="owner"),),
            )
            retained[operation.operation_id] = ProjectReceipt(
                organization_id="org",
                project_id=operation.project_id,
                operation_id=operation.operation_id,
                request_digest_sha256=contract_digest(
                    {
                        "principal": principal.model_dump(mode="json"),
                        "request": operation.model_dump(mode="json"),
                    }
                ),
                state_digest_sha256=contract_digest(state),
                recorded_at=datetime.now(UTC),
                principal=principal,
                state=state,
            ).model_dump(mode="json")
        metadata = {**capability["metadata"], "contract": "core.v1"}
        return httpx.Response(
            200,
            headers=headers,
            json={"metadata": metadata, "receipt": retained[operation.operation_id]},
        )

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
def test_both_clients_keep_original_create_identity_and_normalized_body(async_mode):
    transport, calls = fixture_transport()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            first = await client.forge.projects.create(
                title="Study",
                objective="Compare",
                mutation=MutationOptions(idempotency_key="original-create"),
            )
            again = await client.forge.projects.create(
                title="Study",
                objective="Compare",
                mutation=MutationOptions(idempotency_key="original-create"),
            )
            assert first.receipt == again.receipt

    if async_mode:
        asyncio.run(run())
    else:
        with (
            httpx.Client(transport=transport) as network,
            SynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(http_client=network),
            ) as client,
        ):
            first = client.forge.projects.create(
                title="Study",
                objective="Compare",
                mutation=MutationOptions(idempotency_key="original-create"),
            )
            again = client.forge.projects.create(
                title="Study",
                objective="Compare",
                mutation=MutationOptions(idempotency_key="original-create"),
            )
            assert first.receipt == again.receipt
    assert len(calls) == 2 and calls[0].content == calls[1].content


def test_forge_mcp_inputs_cannot_select_original_identity_and_skill_is_exact():
    tool = BY_NAME["forge_project_create"]
    with pytest.raises(ValueError):
        tool.validate_arguments(
            {
                "title": "Study",
                "objective": "Compare",
                "idempotency_key": "create",
                "credential_id": "other",
            }
        )
    skill = research_skill()
    assert skill.sha256 == hashlib.sha256(skill.content.encode()).hexdigest()
    assert skill.uri.endswith(skill.sha256)
    assert tool.scope == "forge:write" and not tool.paid
