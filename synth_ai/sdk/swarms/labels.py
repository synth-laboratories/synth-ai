"""Explicit compare-and-set labels; metadata never changes run completion."""

import json
import math

from synth_ai.contracts import swarms as operations
from synth_ai.core.parameters import verified
from synth_ai.sdk.core.artifact_types import ArtifactLabelsUpdate
from synth_ai.types.base import ResponseModel
from .types.runs import RunSnapshot


class RunLabels:
    def __init__(self, run):
        self.run = run

    def get(self, *, request=None):
        return self.run.status(request=request).labels

    def update(self, *, set=None, remove=(), expected_revision, mutation=None):
        body = ArtifactLabelsUpdate(
            set=set or {}, remove=tuple(remove), expected_revision=expected_revision
        )
        return self.run._transport.request(
            operations.LABELS,
            RunSnapshot,
            path=f"/swarms/runs/{self.run.run_id}/labels",
            body=body,
            options=mutation,
            validate=self.run._observe_snapshot,
        )

    def set(self, *, mutation=None, **values):
        snapshot = self.run.status()
        return self.update(
            set=label_values(values), expected_revision=snapshot.labels_revision, mutation=mutation
        )


class AsyncRunLabels:
    def __init__(self, run):
        self.run = run

    async def get(self, *, request=None):
        return (await self.run.status(request=request)).labels

    async def update(self, *, set=None, remove=(), expected_revision, mutation=None):
        body = ArtifactLabelsUpdate(
            set=set or {}, remove=tuple(remove), expected_revision=expected_revision
        )
        return await self.run._transport.request(
            operations.LABELS,
            RunSnapshot,
            path=f"/swarms/runs/{self.run.run_id}/labels",
            body=body,
            options=mutation,
            validate=self.run._observe_snapshot,
        )

    async def set(self, *, mutation=None, **values):
        snapshot = await self.run.status()
        return await self.update(
            set=label_values(values), expected_revision=snapshot.labels_revision, mutation=mutation
        )


def label_values(values):
    result = {}
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
