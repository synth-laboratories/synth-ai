"""Inspect a retained isolation gate and its native driver receipts offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .isolation import RESOURCE_NOT_FOUND

DRIVERS = {"sdk", "cli", "mcp_stdio"}


def inspect_cell(row: Mapping[str, Any]) -> str:
    """Keep the producer verdict. Reject a pass its retained summaries contradict."""
    operation = row["operation"]
    result = row["result"]
    if not isinstance(operation, str) or not operation or not isinstance(result, str):
        raise ValueError("invalid_retained_cell")
    if result != "pass":
        return result
    foreign, absent = row["foreign"], row["absent"]
    for prefix in ("org", "user"):
        owner = row.get(f"owner_{prefix}_id")
        other = row.get(f"foreign_{prefix}_id")
        if not owner or not other or owner == other:
            raise ValueError("retained_pass_unsubstantiated")
    for sample in (foreign, absent):
        if (
            sample["accepted"] is not False
            or sample["typed"] is not True
            or sample["status"] != 404
            or sample["code"] not in RESOURCE_NOT_FOUND
            or sample["operation"] != operation
        ):
            raise ValueError("retained_pass_unsubstantiated")
    if foreign["code"] != absent["code"] or foreign.get("cause") != absent.get("cause"):
        raise ValueError("retained_pass_unsubstantiated")
    return result


def inspect_gate(path: Path, manifest_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    gate = json.loads(raw)
    if (
        gate["check"] != "cross_org_isolation"
        or gate["cohort"]["manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("wrong_retained_gate_or_cohort")
    drivers = {}
    for artifact in gate["artifacts"]:
        receipt_path = Path(artifact["path"]).resolve()
        if receipt_path.name != "driver-receipt.json":
            continue
        if not receipt_path.is_relative_to(path.resolve().parent):
            raise ValueError("driver_receipt_outside_gate")
        encoded = receipt_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != artifact["sha256"]:
            raise ValueError("retained_driver_digest_mismatch")
        receipt = json.loads(encoded)
        driver = receipt["driver"]
        if driver not in DRIVERS or driver in drivers or receipt_path.parent.name != driver:
            raise ValueError("retained_driver_identity_mismatch")
        steps = [s for s in receipt["steps"] if s["step"] == "cross_org_not_found_matrix"]
        if len(steps) != 1 or not steps[0]["evidence"]:
            raise ValueError("missing_retained_matrix")
        cells = {row["operation"]: inspect_cell(row) for row in steps[0]["evidence"]}
        if len(cells) != len(steps[0]["evidence"]):
            raise ValueError("duplicate_retained_operation")
        drivers[driver] = {
            "sha256": artifact["sha256"],
            "counts": dict(Counter(cells.values())),
            "cells": cells,
        }
    if set(drivers) != DRIVERS or len({frozenset(d["cells"]) for d in drivers.values()}) != 1:
        raise ValueError("retained_driver_coverage_mismatch")
    return {
        "manifest_sha256": manifest_sha256,
        "gate_sha256": hashlib.sha256(raw).hexdigest(),
        "producer_verdict": gate["result"]["verdict"],
        "drivers": drivers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_gate(args.gate, args.manifest_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
