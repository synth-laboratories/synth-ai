"""Guidance inspires planner-authored objectives; only independent verdicts verify them."""

from typing import Literal
from pydantic import Field, StrictInt, model_serializer
from synth_ai.types.base import Identifier, Model, ResponseBase
from synth_ai.types.refs import Ref


class OmitAbsent:
    @model_serializer(mode="wrap")
    def omit_absent(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


class Guidance(OmitAbsent, Model):
    id: Identifier
    kind: Literal["question", "outcome", "any"] = "any"
    text: str = Field(min_length=1, max_length=8192)
    evidence: str | None = Field(default=None, max_length=8192)
    acceptance: str | None = Field(default=None, max_length=8192)
    priority: Literal["must", "should", "nice"] = "should"


class ObjectiveVerdict(OmitAbsent, ResponseBase):
    id: str
    decision: Literal["verified", "rejected", "partial"]
    reason: str
    evidence: tuple[Ref, ...] = Field(max_length=64)
    by: str
    role: Literal["reviewer"]
    score: StrictInt | None = Field(default=None, ge=0, le=10000)
    claim_ids: tuple[str, ...] = ()


class ObjectiveAnswer(OmitAbsent, ResponseBase):
    revision: StrictInt = Field(ge=1)
    text: str
    sources: tuple[Ref, ...] = Field(max_length=64)
    by: str


class ObjectiveResult(OmitAbsent, ResponseBase):
    id: str
    kind: Literal["oeq", "deo"]
    title: str
    statement: str
    required: bool
    criteria: tuple[str, ...]
    definition_revision: StrictInt = Field(ge=1)
    derived_from: tuple[str, ...] = ()
    state: Literal["open", "claimed", "verified", "rejected", "partial", "not_applicable"]
    verdict: ObjectiveVerdict | None = None
    answer: ObjectiveAnswer | None = None
    claims: tuple[dict, ...] = Field(default=(), max_length=1024)


class GuidanceCoverage(OmitAbsent, ResponseBase):
    guidance_id: str
    state: Literal["covered", "declined"]
    objectives: tuple[str, ...] = ()
    reason: str | None = None
    by: str
