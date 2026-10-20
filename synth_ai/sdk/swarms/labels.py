"""Explicit compare-and-set labels; metadata never changes run completion."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from synth_ai.contracts import swarms as operations
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.sdk.core.artifact_types import ArtifactLabelsUpdate

from .types.runs import RunSnapshot

if TYPE_CHECKING:
    from .run import AsyncSwarmRun, SwarmRun


class RunLabels:
    def __init__(self, run: SwarmRun) -> None:
        self.run = run

    def get(self, *, request: ReadOptions | None = None) -> dict[str, str]:
        return dict(self.run.status(request=request).labels)

    def update(
        self,
        *,
        set: Mapping[str, str] | None = None,
        remove: tuple[str, ...] | list[str] = (),
        expected_revision: int,
        mutation: MutationOptions | None = None,
    ) -> RunSnapshot:
        body = ArtifactLabelsUpdate(
            set=dict(set or {}), remove=tuple(remove), expected_revision=expected_revision
        )
        return self.run._transport.request(
            operations.LABELS,
            RunSnapshot,
            path=f"/swarms/runs/{self.run.run_id}/labels",
            body=body,
            options=mutation,
            validate=self.run._observe_snapshot,
        )

    def set(self, *, mutation: MutationOptions | None = None, **values: object) -> RunSnapshot:
        snapshot = self.run.status()
        return self.update(
            set=label_values(values), expected_revision=snapshot.labels_revision, mutation=mutation
        )


class AsyncRunLabels:
    def __init__(self, run: AsyncSwarmRun) -> None:
        self.run = run

    async def get(self, *, request: ReadOptions | None = None) -> dict[str, str]:
        return dict((await self.run.status(request=request)).labels)

    async def update(
        self,
        *,
        set: Mapping[str, str] | None = None,
        remove: tuple[str, ...] | list[str] = (),
        expected_revision: int,
        mutation: MutationOptions | None = None,
    ) -> RunSnapshot:
        body = ArtifactLabelsUpdate(
            set=dict(set or {}), remove=tuple(remove), expected_revision=expected_revision
        )
        return await self.run._transport.request(
            operations.LABELS,
            RunSnapshot,
            path=f"/swarms/runs/{self.run.run_id}/labels",
            body=body,
            options=mutation,
            validate=self.run._observe_snapshot,
        )

    async def set(
        self, *, mutation: MutationOptions | None = None, **values: object
    ) -> RunSnapshot:
        snapshot = await self.run.status()
        return await self.update(
            set=label_values(values), expected_revision=snapshot.labels_revision, mutation=mutation
        )


def label_values(values: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in values.items():
        if isinstance(value, str):
            result[key] = value
        elif isinstance(value, (int, float, bool)) and (
            not isinstance(value, float) or math.isfinite(value)
        ):
            result[key] = json.dumps(value, allow_nan=False, separators=(",", ":"))
        else:
            raise ValueError("Label values must be strings or finite JSON scalar numbers/booleans")
    return result
