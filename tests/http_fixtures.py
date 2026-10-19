"""Offline public-client exchanges with actual capability negotiation."""

from __future__ import annotations

import httpx
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts.fingerprint import schema_sha256
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.types.capabilities import Capabilities, CapabilitySnapshot, snapshot_sha256
from tests.test_index_retention import METADATA, options


def capabilities_body():
    snapshot = {
        "schema_version": "synth.capabilities.v1",
        "release": {
            "environment": "development",
            "release_train": "0.23.3",
            "candidate_id": "offline-fixture",
            "manifest_sha256": "a" * 64,
            "components": [
                {
                    "component_id": "backend",
                    "version": "0.23.3",
                    "source_commit": "b" * 40,
                    "artifact_sha256": "c" * 64,
                    "configuration_sha256": "d" * 64,
                }
            ],
        },
        "admission": "qualified",
        "contracts": [{"contract": "index.v1", "schema_sha256": schema_sha256("index.v1")}],
        "operations": [
            {
                "operation_id": operation,
                "contract": "index.v1",
                "state": "enabled",
                "surfaces": ["http"],
                "reason": None,
            }
            for operation in ("index.search", "index.searches.list")
        ],
    }
    import json

    digest = snapshot_sha256(CapabilitySnapshot.model_validate_json(json.dumps(snapshot)))
    body = {**snapshot, "metadata": METADATA, "capability_sha256": digest}
    return Capabilities.model_validate_json(json.dumps(body)).model_dump(mode="json")


def negotiated_headers():
    receipt = capabilities_body()
    return {
        "X-Synth-Contract": "index.v1",
        "X-Synth-Schema-SHA256": schema_sha256("index.v1"),
        "X-Synth-Capabilities-SHA256": receipt["capability_sha256"],
        "X-Synth-Release-Manifest-SHA256": receipt["release"]["manifest_sha256"],
    }


def page_response():
    return httpx.Response(
        200,
        headers=negotiated_headers(),
        json={
            "metadata": METADATA,
            "items": [],
            "next_cursor": None,
            "complete": True,
        },
    )


class Exchanges:
    def __init__(self, *responses):
        self.responses = responses
        self.requests = []
        self.capability_requests = []

    def __call__(self, request):
        if request.url.path.endswith("/capabilities"):
            self.capability_requests.append(request)
            return httpx.Response(
                200, headers={"X-Synth-Contract": "index.v1"}, json=capabilities_body()
            )
        self.requests.append(request)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


async def invoke(async_mode, handler, *, mutation=False, client_options=None):
    """Exercise the public facade and verify borrowed network-client lifetime."""
    selected = {"read_retry_count": 2, "retry_delay_seconds": 0, **(client_options or {})}
    if async_mode:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as network:
            client = AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network, **selected),
            )
            try:
                if mutation:
                    return await client.index.search(
                        "query", options=options(), mutation=MutationOptions(idempotency_key="key")
                    )
                return await client.index.searches.list(limit=1)
            finally:
                await client.aclose()
                assert not network.is_closed
    with httpx.Client(transport=httpx.MockTransport(handler)) as network:
        client = SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network, **selected),
        )
        try:
            if mutation:
                return client.index.search(
                    "query", options=options(), mutation=MutationOptions(idempotency_key="key")
                )
            return client.index.searches.list(limit=1)
        finally:
            client.close()
            assert not network.is_closed
