"""Concurrent first use of schema_sha256 must yield one digest (concurrent-create schema_mismatch)."""

from __future__ import annotations

import threading
import time

import pytest

from synth_ai.contracts.fingerprint import SchemaFingerprints, schema_sha256
from synth_ai.core.capabilities import require_operation
from synth_ai.errors import ContractError
from synth_ai.contracts.operations import ALL_OPERATIONS
from synth_ai.types.capabilities import (
    Capabilities,
    ContractIdentity,
    ContractName,
    OperationCapability,
)


class RacyExporter:
    """Stands in for pydantic's cold export: concurrent runs lose a definition."""

    def __init__(self) -> None:
        self.active = 0
        self.calls = 0
        self._guard = threading.Lock()

    def __call__(self, contract: ContractName) -> str:
        with self._guard:
            self.active += 1
            self.calls += 1
            overlapped = self.active > 1
        time.sleep(0.05)
        with self._guard:
            overlapped = overlapped or self.active > 1
            self.active -= 1
        defs = '{"A":1}' if not overlapped else "{}"
        return '{"status":"x","models":' + defs + "}"


def test_concurrent_first_use_serializes_export_and_agrees():
    exporter = RacyExporter()
    fingerprints = SchemaFingerprints(exporter)
    results: list[str] = []
    barrier = threading.Barrier(8)

    def run() -> None:
        barrier.wait()
        results.append(fingerprints.digest("index.v1"))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8
    assert len(set(results)) == 1
    assert exporter.calls == 1


def test_real_fingerprint_is_stable_across_threads():
    results: list[str] = []
    threads = [
        threading.Thread(target=lambda: results.append(schema_sha256("index.v1"))) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1


def test_schema_mismatch_error_names_contract_and_both_hashes():
    operation = ALL_OPERATIONS[0]
    stale = "0" * 64
    receipt = Capabilities.model_construct(
        admission="qualified",
        release=None,
        capability_sha256="1" * 64,
        contracts=(
            ContractIdentity.model_construct(contract=operation.contract, schema_sha256=stale),
        ),
        operations=(
            OperationCapability.model_construct(
                operation_id=operation.id,
                contract=operation.contract,
                state="enabled",
                surfaces=("http",),
                reason=None,
            ),
        ),
    )
    with pytest.raises(ContractError) as raised:
        require_operation(receipt, operation, surface="http")
    error = raised.value
    assert error.code == "schema_mismatch"
    assert error.cause is not None
    assert f"contract={operation.contract}" in error.cause
    assert f"backend_schema_sha256={stale}" in error.cause
    assert f"sdk_schema_sha256={schema_sha256(operation.contract)}" in error.cause
