"""Default tolerance and explicit/pinned strictness through both public clients."""

import json
import logging

import httpx
import pytest
from synth_ai import __version__
from synth_ai.core import capabilities
from synth_ai.errors import ContractError, OutcomeUncertainError
from synth_ai.options import ClientOptions
from synth_ai.types.capabilities import Capabilities, CapabilitySnapshot, snapshot_sha256
from tests.http_fixtures import (
    Exchanges,
    capabilities_body,
    invoke,
    negotiated_headers,
    page_response,
)
from tests.test_index_retention import LEGACY_ZDR, result_body

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def skewed_receipt(*, minimum=None):
    body = capabilities_body()
    body["contracts"][0]["schema_sha256"] = "e" * 64
    if minimum is not None:
        body["contracts"][0]["min_sdk_version"] = minimum
    snapshot = {k: v for k, v in body.items() if k not in {"metadata", "capability_sha256"}}
    body["capability_sha256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(json.dumps(snapshot))
    )
    return body


class ReceiptExchanges(Exchanges):
    def __init__(self, receipt, *responses):
        super().__init__(*responses)
        self.receipt = receipt

    def __call__(self, request):
        if request.url.path.endswith("/capabilities"):
            self.capability_requests.append(request)
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=self.receipt)
        return super().__call__(request)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_schema_skew_is_advisory_once_and_response_hash_checks_follow_mode(
    monkeypatch, caplog, async_mode
):
    monkeypatch.setattr(capabilities, "_SCHEMA_WARNINGS", set())
    body = skewed_receipt()
    with caplog.at_level(logging.WARNING, logger="synth_ai.core.capabilities"):
        for _ in range(2):
            handler = ReceiptExchanges(body, page_response())
            result = await invoke(async_mode, handler)
            assert result.complete
            assert handler.requests[0].headers["X-Synth-Schema-SHA256"] == "e" * 64
    warnings = [r for r in caplog.records if r.name == "synth_ai.core.capabilities"]
    assert len(warnings) == 1 and "tolerant response decoding" in warnings[0].message


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("minimum", [None, "0.24.0"])
async def test_strict_schema_refusal_names_versions_hashes_and_upgrade(async_mode, minimum):
    handler = ReceiptExchanges(skewed_receipt(minimum=minimum), page_response())
    with pytest.raises(ContractError) as caught:
        await invoke(async_mode, handler, client_options={"strict_contract": True})
    error = caught.value
    assert error.code == "schema_mismatch" and error.stage == "capability_negotiation"
    assert error.operation == "index.searches.list" and handler.requests == []
    for value in [
        __version__,
        "release 0.23.3",
        "eeeeeeeeeeee",
        "Upgrade synth-ai",
        "pip install -U synth-ai",
    ]:
        assert value in str(error)
    if minimum is not None:
        assert ">=0.24.0" in str(error)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("pin", ["release_manifest_sha256", "candidate_manifest_sha256"])
async def test_release_and_candidate_pins_imply_strictness(async_mode, pin):
    selected = {pin: "a" * 64, "strict_contract": False}
    assert ClientOptions(**selected).strict_contract is True
    handler = ReceiptExchanges(skewed_receipt(), page_response())
    with pytest.raises(ContractError) as caught:
        await invoke(async_mode, handler, client_options=selected)
    assert caught.value.code == "schema_mismatch" and not handler.requests


@pytest.mark.parametrize("async_mode", [False, True])
async def test_strict_response_identity_refusal_includes_upgrade_context(async_mode):
    headers = {**negotiated_headers(), "X-Synth-Schema-SHA256": "e" * 64}
    handler = Exchanges(httpx.Response(200, headers=headers, json={"invalid": "body"}))
    with pytest.raises(ContractError) as caught:
        await invoke(async_mode, handler, client_options={"strict_contract": True})
    assert caught.value.code == "schema_mismatch" and caught.value.stage == "response_decode"
    assert "Upgrade synth-ai" in str(caught.value) and __version__ in str(caught.value)
    assert len(handler.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_malformed_known_response_still_reports_schema_versions(async_mode):
    handler = Exchanges(
        httpx.Response(
            200,
            headers=negotiated_headers(),
            json={
                "metadata": capabilities_body()["metadata"],
                "items": "not an array",
                "next_cursor": None,
                "complete": True,
            },
        )
    )
    with pytest.raises(ContractError) as caught:
        await invoke(async_mode, handler)
    assert caught.value.code == "schema_mismatch" and caught.value.stage == "response_decode"
    assert "release 0.23.3" in str(caught.value) and "Upgrade synth-ai" in str(caught.value)
    assert len(handler.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_extra_mutation_response_fields_succeed_without_false_uncertainty(async_mode):
    body = result_body(LEGACY_ZDR)
    body["future_response_field"] = True
    body["retention"] = {**LEGACY_ZDR, "future_nested_field": True}
    handler = Exchanges(httpx.Response(200, headers=negotiated_headers(), json=body))
    result = await invoke(async_mode, handler, mutation=True)
    assert result.response == "answer" and len(handler.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_malformed_mutation_response_retains_custody_and_upgrade_diagnostic(async_mode):
    handler = Exchanges(httpx.Response(200, headers=negotiated_headers(), json={"broken": True}))
    with pytest.raises(OutcomeUncertainError) as caught:
        await invoke(async_mode, handler, mutation=True)
    assert caught.value.reason == "response_invalid" and caught.value.identity.key == "key"
    assert "Upgrade synth-ai" in str(caught.value) and len(handler.requests) == 1


async def test_optional_minimum_sdk_version_preserves_old_receipt_digest_when_absent():
    old = capabilities_body()
    parsed = Capabilities.model_validate_json(json.dumps(old))
    assert "min_sdk_version" not in parsed.model_dump(mode="json")["contracts"][0]
    new = skewed_receipt(minimum="0.24.0")
    assert (
        Capabilities.model_validate_json(json.dumps(new)).contracts[0].min_sdk_version == "0.24.0"
    )
