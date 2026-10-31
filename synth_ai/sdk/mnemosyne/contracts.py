"""Shared core.v1 evidence facade envelope."""

from pydantic import JsonValue

from synth_ai.types.base import ResponseModel


class EvaluationEvidenceResponse(ResponseModel):
    data: dict[str, JsonValue]
