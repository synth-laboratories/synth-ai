"""Negotiated project evidence adapter tests with a simulated customer facade."""

import asyncio
import hashlib
import json

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ContractError, UnsupportedOperationError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

PROJECT = "81000000-0000-4000-8000-000000000002"
ORG = "81000000-0000-4000-8000-000000000001"
OP = "op-1"
EVIDENCE = (
    "mev_"
    + hashlib.sha256(json.dumps([ORG, PROJECT, OP], separators=(",", ":")).encode()).hexdigest()
)


def mock_wire(*, advertise=True, changed=False, wrong_org=False):
    from synth_ai.contracts.fingerprint import schema_sha256

    capability, headers = receipt_and_headers()
    capability["contracts"].append(
        {"contract": "index.v1", "schema_sha256": schema_sha256("index.v1")}
    )
    capability["operations"].append(
        {
            "operation_id": "auth.context",
            "contract": "index.v1",
            "state": "enabled",
            "surfaces": ["http", "mcp"],
            "reason": None,
        }
    )
    if advertise:
        for suffix in ("capabilities", "list", "get", "measurements"):
            capability["operations"].append(
                {
                    "operation_id": f"mnemosyne.evidence.{suffix}",
                    "contract": "core.v1",
                    "state": "enabled",
                    "surfaces": ["http", "mcp"],
                    "reason": None,
                }
            )
    capability["operations"].sort(key=lambda x: x["operation_id"])
    snapshot = {k: v for k, v in capability.items() if k not in {"metadata", "capability_sha256"}}
    capability["capability_sha256"] = headers["X-Synth-Capabilities-SHA256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(json.dumps(snapshot))
    )
    calls = []

    def respond(request):
        if request.url.path == "/api/v1/capabilities":
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        if request.url.path == "/api/v1/auth/context":
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "index.v1"},
                json={
                    "metadata": capability["metadata"],
                    "tenant_id": "other-org" if wrong_org else ORG,
                    "subject_id": "user",
                    "credential_id": "credential",
                    "credential_kind": "api_key",
                    "oauth_client_id": None,
                    "subject_kind": "user",
                    "scopes": ["forge:read"],
                },
            )
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer fixture"
        path = request.url.path
        if path.endswith("evaluation-evidence-capabilities"):
            data = {
                "schema_version": "synth.mnemosyne.evaluation-evidence-capabilities.v1",
                "scope": {"organization_id": ORG, "project_id": PROJECT},
                "can_read": True,
                "can_publish": False,
                "intake": "unavailable",
                "request_schema_version": "synth.mnemosyne.evaluation-evidence-request.v1",
            }
        elif path.endswith("evaluation-evidence"):
            data = {
                "schema_version": "synth.mnemosyne.evaluation-evidence-page.v1",
                "scope": {"organization_id": ORG, "project_id": PROJECT},
                "items": [],
                "next_cursor": None,
            }
        elif path.endswith("/measurements"):
            data = {
                "evidence_id": EVIDENCE,
                "request_digest": "a" * 64,
                "scope": {"organization_id": ORG, "project_id": PROJECT},
                "source_association": {
                    "source_kind": "resinbench",
                    "verification": "producer_asserted",
                    "producer_run_id": "run-1",
                    "attempt_id": "attempt-1",
                },
                "summary": {
                    "schema_version": "synth.mnemosyne.measurement-summary.v1",
                    "recipe_version": "evaluation-measurements.v1",
                    "canonical_measurements_sha256": "d" * 64,
                    "producer_run_id": "different" if changed else "run-1",
                    "attempt_id": "attempt-1",
                    "coverage": "partial",
                    "costs": [],
                    "subtotals": [],
                    "spans": [],
                    "run_wall_seconds": "1.234",
                    "limitations": ["missing cost"],
                },
            }
        else:
            data = {
                "schema_version": "synth.mnemosyne.evaluation-evidence-receipt.v1",
                "evidence_id": EVIDENCE,
                "scope": {"organization_id": ORG, "project_id": "other" if changed else PROJECT},
                "operation_id": OP,
                "request_digest": "a" * 64,
                "source_association": {
                    "source_kind": "resinbench",
                    "verification": "producer_asserted",
                },
                "objects": [
                    {
                        "role": "report",
                        "locator": {
                            "schema_version": "synth.forge.evidence.v1",
                            "result": "present",
                            "scope": {"organization_id": ORG, "project_id": PROJECT},
                            "digest_sha256": "b" * 64,
                            "size_bytes": 3,
                            "media_type": "application/json",
                            "manifest": {
                                "authority": "artifact-platform",
                                "kind": "manifest",
                                "record_id": "publication",
                                "revision": "1",
                                "digest_sha256": "c" * 64,
                            },
                            "object": {
                                "authority": "artifact-platform",
                                "kind": "object",
                                "record_id": "publication",
                                "revision": "1",
                                "digest_sha256": "b" * 64,
                            },
                            "delivery": {
                                "publication_id": "81000000-0000-4000-8000-000000000003",
                                "logical_path": "sha256/" + "b" * 64,
                            },
                        },
                    }
                ],
                "imported_by": "user:user",
                "custody": "artifact_reference",
            }
        return httpx.Response(
            200,
            headers=headers,
            json={"metadata": {**capability["metadata"], "contract": "core.v1"}, "data": data},
        )

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
def test_sdk_calls_capabilities_list_and_get_with_caller_auth(async_mode):
    transport, calls = mock_wire()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            assert not (
                await client.mnemosyne.evaluation_evidence.capabilities(PROJECT)
            ).data.can_publish
            assert not (
                await client.mnemosyne.evaluation_evidence.list(PROJECT, limit=7)
            ).data.items
            tool = BY_NAME["mnemosyne_evaluation_evidence_get"]
            result = await tool.invoke(client, {"project_id": PROJECT, "evidence_id": EVIDENCE})
            assert result["data"]["source_association"]["verification"] == "producer_asserted"

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
            assert not client.mnemosyne.evaluation_evidence.capabilities(PROJECT).data.can_publish
            assert not client.mnemosyne.evaluation_evidence.list(PROJECT, limit=7).data.items
            assert (
                client.mnemosyne.evaluation_evidence.get(PROJECT, EVIDENCE).data.evidence_id
                == EVIDENCE
            )
    assert len(calls) == 3
    assert calls[1].url.params["limit"] == "7"


def test_capability_absence_refuses_before_evidence_request():
    transport, calls = mock_wire(advertise=False)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
        pytest.raises(UnsupportedOperationError),
    ):
        client.mnemosyne.evaluation_evidence.get(PROJECT, EVIDENCE)
    assert calls == []


def test_changed_project_refuses_receipt():
    transport, _ = mock_wire(changed=True)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
        pytest.raises(ContractError),
    ):
        client.mnemosyne.evaluation_evidence.get(PROJECT, EVIDENCE)


def test_receipt_must_match_current_authenticated_tenant():
    transport, _ = mock_wire(wrong_org=True)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
        pytest.raises(ContractError),
    ):
        client.mnemosyne.evaluation_evidence.get(PROJECT, EVIDENCE)


@pytest.mark.parametrize("changed", [False, True])
def test_measurements_preserve_partial_coverage_and_source_identity(changed):
    from decimal import Decimal

    transport, _ = mock_wire(changed=changed)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
    ):
        if changed:
            with pytest.raises(ContractError):
                client.mnemosyne.evaluation_evidence.measurements(PROJECT, EVIDENCE)
        else:
            summary = client.mnemosyne.evaluation_evidence.measurements(
                PROJECT, EVIDENCE
            ).data.summary
            assert summary.run_wall_seconds == Decimal("1.234")
            assert summary.coverage == "partial" and summary.limitations == ("missing cost",)


def test_mcp_inputs_refuse_authority_and_validate_cursor():
    tool = BY_NAME["mnemosyne_evaluation_evidence_list"]
    assert tool.scope == "forge:read" and not tool.mutates and not tool.paid
    with pytest.raises(ValueError):
        tool.validate_arguments({"project_id": PROJECT, "principal": "invented"})
    from synth_ai.sdk.mnemosyne.evidence import page_query

    with pytest.raises(ValueError):
        page_query("wrong", 10)


@pytest.mark.parametrize(
    "advertise,scope,expected", [(False, True, False), (True, False, False), (True, True, True)]
)
def test_mcp_discovery_needs_backend_capability_and_caller_scope(advertise, scope, expected):
    from types import SimpleNamespace

    from synth_ai.mcp.runtime import SDKExecutor
    from synth_ai.types.capabilities import Capabilities

    network, _ = mock_wire(advertise=advertise)
    capability = network.handler(httpx.Request("GET", "https://backend.test/api/v1/capabilities"))
    receipt = Capabilities.model_validate_json(capability.content)

    class Client:
        base_url = "https://backend.test"
        candidate_manifest_sha256 = None
        _transport = SimpleNamespace(strict_contract=False)

        async def capabilities(self, **kwargs):
            return receipt

        async def auth_context(self, **kwargs):
            return SimpleNamespace(scopes=("forge:read",) if scope else ())

    selected = frozenset({"mnemosyne_evaluation_evidence_get"})
    result = asyncio.run(SDKExecutor(Client(), Client(), selected).available_tools())
    assert ("mnemosyne_evaluation_evidence_get" in result) is expected


@pytest.mark.parametrize(
    "advertise,scope,allowed,expected",
    [
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_hosted_mcp_discovery_needs_backend_scope_and_principal(
    monkeypatch, advertise, scope, allowed, expected
):
    from types import SimpleNamespace

    pytest.importorskip("mcp")
    from synth_ai.mcp.hosted import HostedExecutor
    from synth_ai.types.capabilities import Capabilities

    network, _ = mock_wire(advertise=advertise)
    response = network.handler(httpx.Request("GET", "https://backend.test/api/v1/capabilities"))
    receipt = Capabilities.model_validate_json(response.content)
    tool = "mnemosyne_evaluation_evidence_get"
    principal = SimpleNamespace(
        scopes=("forge:read",) if scope else (), allowed_tools={tool} if allowed else set()
    )
    monkeypatch.setattr(HostedExecutor, "principal", lambda self: principal)
    executor = HostedExecutor(None, frozenset({tool}), lambda: receipt, None)
    assert (tool in asyncio.run(executor.available_tools())) is expected
