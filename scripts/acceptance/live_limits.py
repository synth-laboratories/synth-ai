"""Raise four public limits on an owned running run and retain each budget view.

See SYN-5663. Call from the cached journey while execution is running. This
scenario never launches a run or selects a target; its caller supplies the
authorized client and explicit ceilings. A retained intent refuses a rerun,
including after an uncertain response, until its original identity is reconciled.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from synth_ai.models import RunLimitChanges, RunLimitsProgress, RunLimitUpdate

if TYPE_CHECKING:
    from pathlib import Path

    from synth_ai import SynthClient

__all__ = ["LimitTargets", "LiveLimitsError", "contract_maximum", "raise_live_limits"]


class Dimension(StrEnum):
    SPEND = "spend_micros"
    TOKENS = "tokens"
    EXECUTION = "execution_ms"
    RETENTION = "pause_retention_ms"


def contract_maximum(field: str) -> int:
    """The public contract's inclusive maximum for one RunLimitChanges integer field."""
    schema = RunLimitChanges.model_json_schema()["properties"][field]
    bounds = [branch["maximum"] for branch in schema.get("anyOf", [schema]) if "maximum" in branch]
    if len(bounds) != 1:
        raise ValueError(f"contract maximum unavailable: {field}")
    return int(bounds[0])


@dataclass(frozen=True, slots=True)
class LimitTargets:
    """Raise targets. A ``None`` seconds target means the cap already sits at the
    contract maximum: that dimension is not mutated, and every following budget
    view must still show it unchanged.
    """

    budget_usd: str
    maximum_tokens: int
    wall_time_seconds: int | None
    pause_retention_seconds: int | None

    def held(self) -> tuple[Dimension, ...]:
        return tuple(
            dimension
            for dimension, value in (
                (Dimension.EXECUTION, self.wall_time_seconds),
                (Dimension.RETENTION, self.pause_retention_seconds),
            )
            if value is None
        )

    def changes(self) -> tuple[tuple[Dimension, int, RunLimitChanges], ...]:
        charge = Decimal(self.budget_usd)
        micros = charge * 1_000_000
        if not charge.is_finite() or charge <= 0 or micros != micros.to_integral_value():
            raise ValueError("budget must be a positive whole number of microdollars")
        from synth_ai.models import Money

        rows = (
            (
                Dimension.SPEND,
                int(micros),
                RunLimitChanges(maximum_charge=Money(amount=self.budget_usd, currency="USD")),
            ),
            (
                Dimension.TOKENS,
                self.maximum_tokens,
                RunLimitChanges(maximum_tokens=self.maximum_tokens),
            ),
        )
        if self.wall_time_seconds is not None:
            rows += (
                (
                    Dimension.EXECUTION,
                    self.wall_time_seconds * 1000,
                    RunLimitChanges(wall_time_seconds=self.wall_time_seconds),
                ),
            )
        if self.pause_retention_seconds is not None:
            rows += (
                (
                    Dimension.RETENTION,
                    self.pause_retention_seconds * 1000,
                    RunLimitChanges(pause_retention_seconds=self.pause_retention_seconds),
                ),
            )
        return rows


class LiveLimitsError(RuntimeError):
    def __init__(self, code: str, *, run_id: str, stage: str) -> None:
        self.code = code
        self.run_id = run_id
        self.stage = stage
        super().__init__(f"{code}: run={run_id} stage={stage}")


def retain(path: Path, payload: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(payload + "\n")
        output.flush()
        os.fsync(output.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def caps(view: RunLimitsProgress, run_id: str) -> dict[Dimension, int]:
    if view.run_id != run_id or view.status != "running":
        raise LiveLimitsError("run_not_running", run_id=run_id, stage="budget_view")
    values: dict[Dimension, int] = {}
    for dimension in Dimension:
        rows = [row for row in view.primary if row.dimension == dimension]
        if len(rows) != 1:
            raise LiveLimitsError("limit_dimension_invalid", run_id=run_id, stage=dimension)
        row = rows[0]
        # Core primaryCaps uses requested_cap for metered-only tokens. A null
        # effective cap is intentional, not permission to fabricate one.
        cap = (
            row.requested_cap
            if dimension == Dimension.TOKENS
            and row.effective_cap is None
            and row.state == "metered"
            else row.effective_cap
        )
        if cap is None:
            raise LiveLimitsError("limit_cap_unavailable", run_id=run_id, stage=dimension)
        values[dimension] = cap
    return values


def raise_live_limits(
    client: SynthClient,
    *,
    run_id: str,
    targets: LimitTargets,
    mutation_prefix: str,
    receipt_directory: Path,
) -> tuple[RunLimitsProgress, ...]:
    """Assert each change in the immediately following authoritative budget view.

    The exclusive receipt directory prevents accidental repeat execution. On an
    uncertain mutation, retain its intent and reconcile it; do not call this
    function again with a fresh directory to force another update.
    """
    if not mutation_prefix or len(mutation_prefix) > 100:
        raise ValueError("mutation_prefix must contain 1..100 characters")
    changes = targets.changes()  # Validate all targets before any mutation.
    receipt_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    current = client.swarms.limits(run_id)
    retain(receipt_directory / "before.json", current.model_dump_json())
    expected_caps = caps(current, run_id)
    for dimension, target, _ in changes:
        if target <= expected_caps[dimension]:
            raise LiveLimitsError("limit_target_not_increased", run_id=run_id, stage=dimension)
    maxima = {
        Dimension.EXECUTION: contract_maximum("wall_time_seconds") * 1000,
        Dimension.RETENTION: contract_maximum("pause_retention_seconds") * 1000,
    }
    for dimension in targets.held():
        # Holding is only legal when the contract forbids any raise.
        if expected_caps[dimension] < maxima[dimension]:
            raise LiveLimitsError("limit_target_not_increased", run_id=run_id, stage=dimension)
    observed: list[RunLimitsProgress] = []
    for index, (dimension, target, change) in enumerate(changes):
        key = f"{mutation_prefix}:{dimension}"
        request = RunLimitUpdate(
            expected_revision=current.revision,
            limits=change,
            resume=False,
            reason="SYN-5663 cached acceptance",
        )
        retain(
            receipt_directory / f"{index}-intent.json",
            json.dumps(
                {
                    "run_id": run_id,
                    "idempotency_key": key,
                    "request": request.model_dump(exclude_none=True),
                },
                sort_keys=True,
            ),
        )
        response = client.swarms.update_limits_request(run_id, request, idempotency_key=key)
        retain(receipt_directory / f"{index}-response.json", response.model_dump_json())
        following = client.swarms.limits(run_id)
        retain(receipt_directory / f"{index}-view.json", following.model_dump_json())
        expected_caps[dimension] = target
        if (
            response.run_id != run_id
            or response.revision != current.revision + 1
            or following.revision != response.revision
        ):
            raise LiveLimitsError("limit_revision_mismatch", run_id=run_id, stage=dimension)
        if caps(following, run_id) != expected_caps:
            raise LiveLimitsError("limit_update_not_observed", run_id=run_id, stage=dimension)
        observed.append(following)
        current = following
    retain(
        receipt_directory / "verdict.json",
        json.dumps(
            {
                "ticket": "SYN-5663",
                "run_id": run_id,
                "verdict": "Pass",
                "scope": "four public limit raises and next budget views",
                "raised": [dimension.value for dimension, _, _ in changes],
                "held_at_contract_maximum": [dimension.value for dimension in targets.held()],
                "revisions": [view.revision for view in observed],
            },
            sort_keys=True,
        ),
    )
    return tuple(observed)
