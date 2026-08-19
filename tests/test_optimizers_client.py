"""Focused tests for the hosted-optimizer clients.

The surface under test is read-only discovery and lineage: checkpoints, the
hosted-training model catalog, and persisted run outputs. Submission and
cancellation are deliberately absent here -- they are backend-owned and are not
reachable through this client -- so these tests assert what the client actually
does rather than inventing coverage for endpoints it does not call.

Every request is served by an httpx MockTransport, so nothing here needs a
network, a credential, or a running backend.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from synth_ai.core.errors import (
    AuthorizationError,
    ConflictError,
    RateLimitedError,
    ResearchOperationError,
    TransientServiceError,
)
from synth_ai.sdk.optimizers import AsyncOptimizersClient, OptimizersClient

API_KEY = "sk-test-do-not-leak-9f3a"


def _mount(client: Any, handler: Any) -> list[httpx.Request]:
    """Replace the live httpx client with a mock, preserving base URL + headers."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = client._transport
    old = transport.client
    is_async = isinstance(old, httpx.AsyncClient)
    kind = httpx.AsyncClient if is_async else httpx.Client
    mock = httpx.MockTransport(record) if not is_async else httpx.MockTransport(record)
    transport.client = kind(
        base_url=str(old.base_url),
        headers=old.headers,
        transport=mock,
    )
    return seen


def _json(payload: Any, status: int = 200) -> Any:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


CHECKPOINT = {
    "checkpoint_id": "ckpt_1",
    "org_id": "org_1",
    "visibility": "org",
    "name": "banking77-cispo-step-40",
    "provider": "tinker",
    "checkpoint_kind": "lora",
    "base_model": "openai/gpt-oss-20b",
    "status": "ready",
    "optimizer_algorithm": "cispo",
    "run_id": "run_1",
    "storage": {
        "backend": "wasabi",
        "bucket": "synth-checkpoints",
        "key": "org_1/run_1/ckpt_1.safetensors",
        "content_type": "application/octet-stream",
        "sha256": "a" * 64,
    },
    "lineage": {
        "optimizer_algorithm": "cispo",
        "run_id": "run_1",
        "attempt_id": "attempt_1",
        "source_checkpoint_id": "ckpt_0",
        "provider_checkpoint_reference": "tinker://job/abc/step/40",
    },
}


# --- checkpoint and lineage decoding -------------------------------------


def test_checkpoint_lineage_decodes_and_is_traversable() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json(CHECKPOINT))
    ckpt = client.checkpoints.get("ckpt_1")
    assert ckpt.checkpoint_id == "ckpt_1"
    assert ckpt.storage.bucket == "synth-checkpoints"
    # Lineage is the point: a checkpoint must name what produced it.
    assert ckpt.lineage.run_id == "run_1"
    assert ckpt.lineage.attempt_id == "attempt_1"
    assert ckpt.lineage.source_checkpoint_id == "ckpt_0"
    assert ckpt.lineage.provider_checkpoint_reference == "tinker://job/abc/step/40"


def test_checkpoint_without_lineage_block_defaults_to_empty_not_missing() -> None:
    payload = {k: v for k, v in CHECKPOINT.items() if k != "lineage"}
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json(payload))
    ckpt = client.checkpoints.get("ckpt_1")
    # Absent lineage is an empty lineage, not an attribute error at the call site.
    assert ckpt.lineage.run_id is None
    assert ckpt.lineage.source_checkpoint_id is None


def test_run_outputs_decode_identity_artifacts_and_checkpoints() -> None:
    payload = {
        "run": {
            "run_id": "run_1",
            "attempt_id": "attempt_1",
            "optimizer_algorithm": "cispo",
            "status": "succeeded",
        },
        "result": {"meanReward": None},
        "artifacts": [
            {
                "artifact_id": "art_1",
                "run_id": "run_1",
                "artifact_name": "events.jsonl",
                "size_bytes": 2048,
                "storage_backend": "wasabi",
                "uri": "s3://synth/run_1/events.jsonl",
                "download_path": "/api/v1/optimizers/artifacts/art_1",
            }
        ],
        "model_checkpoints": [CHECKPOINT],
        "counts": {"artifacts": 1, "model_checkpoints": 1},
    }
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json(payload))
    outputs = client.runs.outputs("run_1")
    assert outputs.run.optimizer_algorithm == "cispo"
    assert outputs.run.status == "succeeded"
    assert outputs.counts["model_checkpoints"] == 1
    assert outputs.model_checkpoints[0].lineage.attempt_id == "attempt_1"
    # A null aggregate must survive as None, never as 0.0.
    assert outputs.result is not None
    assert outputs.result["meanReward"] is None


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed", "cancelled"])
def test_terminal_and_nonterminal_run_states_decode_verbatim(status: str) -> None:
    payload = {
        "run": {
            "run_id": "run_1",
            "optimizer_algorithm": "sft",
            "status": status,
        },
        "artifacts": [],
        "model_checkpoints": [],
        "counts": {},
    }
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json(payload))
    # The client must not collapse or reinterpret server states.
    assert client.runs.outputs("run_1").run.status == status


# --- request serialization ------------------------------------------------


@pytest.mark.parametrize("algorithm", ["sft", "cispo"])
def test_algorithm_filter_is_serialized_for_both_algorithms(algorithm: str) -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(
        client,
        _json(
            {"catalog_revision": "r1", "live_preflight_required": True, "models": [], "total": 0}
        ),
    )
    client.models.list(algorithm=algorithm)
    assert seen[-1].url.params["algorithm"] == algorithm


def test_empty_and_none_filters_are_omitted_not_sent_blank() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(client, _json({"items": [], "total": 0, "limit": 50, "offset": 0}))
    client.checkpoints.list(optimizer_algorithm=None, run_id="", base_model="gpt-oss")
    params = seen[-1].url.params
    assert "optimizer_algorithm" not in params
    assert "run_id" not in params
    assert params["base_model"] == "gpt-oss"


def test_pagination_is_clamped_to_the_documented_bounds() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(client, _json({"items": [], "total": 0, "limit": 100, "offset": 0}))
    client.checkpoints.list(limit=5000, offset=-10)
    assert seen[-1].url.params["limit"] == "100"
    assert seen[-1].url.params["offset"] == "0"


def test_checkpoint_catalog_mutations_use_owner_scoped_contracts() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(client, _json(CHECKPOINT))
    client.checkpoints.update("ckpt_1", name="reviewed", tags=["reviewed"])
    assert seen[-1].method == "PATCH"
    assert seen[-1].url.path.endswith("/checkpoints/ckpt_1")
    assert json.loads(seen[-1].content) == {"name": "reviewed", "tags": ["reviewed"]}


def test_annotation_contract_carries_real_artifact_ids() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(
        client,
        _json(
            {
                "annotation_id": "note_1",
                "run_id": "run_1",
                "body": "Artifact supports the conclusion.",
                "evidence": [{"artifact_id": "art_1", "artifact_name": "events.jsonl"}],
            }
        ),
    )
    note = client.workbench.annotate(
        run_id="run_1", body="Artifact supports the conclusion.", evidence_artifact_ids=["art_1"]
    )
    assert note.evidence[0]["artifact_id"] == "art_1"
    assert seen[-1].url.path.endswith("/checkpoints/workbench/annotations")
    assert json.loads(seen[-1].content)["evidence_artifact_ids"] == ["art_1"]


def test_run_id_is_path_escaped_and_cannot_traverse() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(
        client,
        _json(
            {
                "run": {"run_id": "x", "optimizer_algorithm": "sft", "status": "queued"},
                "artifacts": [],
                "model_checkpoints": [],
                "counts": {},
            }
        ),
    )
    client.runs.outputs("../../admin/secrets")
    assert "/admin/secrets" not in str(seen[-1].url)
    assert "%2F" in str(seen[-1].url)


# --- authentication -------------------------------------------------------


def test_authorization_header_is_bearer_and_key_stays_out_of_the_url() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    seen = _mount(client, _json({"items": [], "total": 0, "limit": 50, "offset": 0}))
    client.checkpoints.list()
    request = seen[-1]
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert API_KEY not in str(request.url)


def test_error_reporting_does_not_leak_the_credential() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json({"detail": {"error_code": "forbidden"}}, status=403))
    with pytest.raises(AuthorizationError) as caught:
        client.checkpoints.list()
    rendered = f"{caught.value!r} {caught.value}"
    assert API_KEY not in rendered


# --- error normalization --------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (403, AuthorizationError),
        (409, ConflictError),
        (429, RateLimitedError),
        (500, TransientServiceError),
        (503, TransientServiceError),
        (418, ResearchOperationError),
    ],
)
def test_backend_errors_normalize_to_typed_exceptions(status: int, expected: type) -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json({"detail": {"error_code": "boom"}}, status=status))
    with pytest.raises(expected):
        client.checkpoints.list()


def test_normalized_failure_carries_the_backend_error_code_and_retry_hint() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json({"detail": {"error_code": "rate_limited"}}, status=429))
    with pytest.raises(RateLimitedError) as caught:
        client.checkpoints.list()
    failure = caught.value.failure
    assert str(failure.code).endswith("rate_limited") or failure.code == "rate_limited"
    assert failure.retry.retryable is True
    assert failure.status == 429


def test_a_non_object_response_is_refused_rather_than_coerced() -> None:
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json([1, 2, 3]))
    with pytest.raises(ValueError):
        client.checkpoints.list()


# --- forward compatibility ------------------------------------------------


def test_unknown_server_fields_are_preserved_not_dropped() -> None:
    payload = dict(CHECKPOINT)
    payload["quantization"] = "int4"
    payload["storage"] = dict(CHECKPOINT["storage"], region="us-east-2")
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json(payload))
    ckpt = client.checkpoints.get("ckpt_1")
    # extra="allow": a newer server must not break an older client, and the new
    # field has to survive so callers can read it.
    assert ckpt.quantization == "int4"
    assert ckpt.storage.region == "us-east-2"


def test_a_new_algorithm_in_the_catalog_decodes_without_a_client_change() -> None:
    payload = {
        "catalog_revision": "hosted-training-models.2026-08-19.v3",
        "live_preflight_required": True,
        "models": [
            {
                "model_id": "openai/gpt-oss-20b",
                "label": "GPT-OSS 20B",
                "provider": "tinker",
                "provider_revision": "default",
                "architecture": "dense",
                "max_context_length": 131072,
                "rank": {"default": 8, "minimum": 1, "maximum": 4096},
                "algorithms": {
                    "sft": {"status": "available"},
                    "cispo": {
                        "status": "not_validated",
                        "note": "held after the bounded canary; launch remains fail-closed",
                    },
                    "grpo": {"status": "not_validated"},
                },
            }
        ],
        "total": 1,
    }
    client = OptimizersClient(api_key=API_KEY, base_url="https://api.test")
    _mount(client, _json(payload))
    catalog = client.models.list()
    algorithms = catalog.models[0].algorithms
    assert algorithms["cispo"]["status"] == "not_validated"
    assert algorithms["grpo"]["status"] == "not_validated"
    assert catalog.catalog_revision == "hosted-training-models.2026-08-19.v3"


# --- async parity ---------------------------------------------------------


def test_async_client_decodes_identically_to_the_sync_client() -> None:
    async def run() -> Any:
        client = AsyncOptimizersClient(api_key=API_KEY, base_url="https://api.test")
        _mount(client, _json(CHECKPOINT))
        try:
            return await client.checkpoints.get("ckpt_1")
        finally:
            await client.close()

    ckpt = asyncio.run(run())
    assert ckpt.checkpoint_id == "ckpt_1"
    assert ckpt.lineage.attempt_id == "attempt_1"


def test_async_client_normalizes_errors_identically() -> None:
    async def run() -> None:
        client = AsyncOptimizersClient(api_key=API_KEY, base_url="https://api.test")
        _mount(client, _json({"detail": {"error_code": "conflict"}}, status=409))
        try:
            await client.checkpoints.list()
        finally:
            await client.close()

    with pytest.raises(ConflictError):
        asyncio.run(run())
