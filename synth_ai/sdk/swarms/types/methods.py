"""Versioned native research selection. See notes/specifications/sdk/swarms_overrides.md.

These contracts express intent, not deployment capability. Omission preserves
server defaults; unsupported releases must refuse before any paid execution.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self
from pydantic import Field, StrictInt, model_validator
from synth_ai.types.base import Digest, Model, ResponseBase


class ForkFlushRound(Model):
    after_budget_bps: StrictInt = Field(ge=1, le=9999)
    exploration_budget_bps: StrictInt = Field(ge=1, le=9999)


def validate_rounds(rounds: tuple[ForkFlushRound, ...]) -> None:
    previous_end = 0
    for item in rounds:
        end = item.after_budget_bps + item.exploration_budget_bps
        if item.after_budget_bps <= previous_end or end >= 10000:
            raise ValueError(
                "Fork rounds must leave positive exploitation intervals within budget"
            )
        previous_end = end


class ForkFlushParameters(Model):
    branches: StrictInt = Field(ge=2, le=128)
    rounds: tuple[ForkFlushRound, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        validate_rounds(self.rounds)
        return self


class ForkFlushOverrides(Model):
    branches: StrictInt | None = Field(default=None, ge=2, le=128)
    rounds: tuple[ForkFlushRound, ...] | None = Field(
        default=None, min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if self.rounds is not None:
            validate_rounds(self.rounds)
        return self


class ForkFlushMethod(Model):
    release: Literal["research.fork_flush@1"] = "research.fork_flush@1"
    overrides: ForkFlushOverrides | None = None


class ForgettingParameters(Model):
    peers: StrictInt = Field(ge=1, le=128)
    maximum_epochs: StrictInt = Field(ge=1, le=2**32 - 1)
    maximum_tasks: StrictInt = Field(ge=1, le=1000)
    epoch_millis: StrictInt = Field(ge=1, le=2**63 - 1)
    work_spend_micros: StrictInt = Field(ge=1, le=2**63 - 1)
    work_tokens: StrictInt = Field(ge=1, le=2**63 - 1)

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.maximum_tasks < self.peers:
            raise ValueError("Forgetting tasks must cover every peer")
        return self


class ForgettingOverrides(Model):
    peers: StrictInt | None = Field(default=None, ge=1, le=128)
    maximum_epochs: StrictInt | None = Field(default=None, ge=1, le=2**32 - 1)
    maximum_tasks: StrictInt | None = Field(default=None, ge=1, le=1000)
    epoch_millis: StrictInt | None = Field(default=None, ge=1, le=2**63 - 1)
    work_spend_micros: StrictInt | None = Field(default=None, ge=1, le=2**63 - 1)
    work_tokens: StrictInt | None = Field(default=None, ge=1, le=2**63 - 1)


class ForgettingMethod(Model):
    release: Literal["research.forgetting@1"] = "research.forgetting@1"
    overrides: ForgettingOverrides | None = None


class MiraParameters(Model):
    maximum_cycles: StrictInt = Field(ge=1, le=128)
    curated_observations: StrictInt = Field(ge=1, le=64)
    curated_bytes: StrictInt = Field(ge=1024, le=65536)
    planner_spend_micros: StrictInt = Field(ge=1, le=2**64 - 1)
    planner_tokens: StrictInt = Field(ge=1, le=2**64 - 1)
    work_spend_micros: StrictInt = Field(ge=1, le=2**64 - 1)
    work_tokens: StrictInt = Field(ge=1, le=2**64 - 1)
    evaluator_spend_micros: StrictInt = Field(ge=1, le=2**64 - 1)
    evaluator_tokens: StrictInt = Field(ge=1, le=2**64 - 1)


class MiraOverrides(Model):
    maximum_cycles: StrictInt | None = Field(default=None, ge=1, le=128)
    curated_observations: StrictInt | None = Field(default=None, ge=1, le=64)
    curated_bytes: StrictInt | None = Field(default=None, ge=1024, le=65536)
    planner_spend_micros: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    planner_tokens: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    work_spend_micros: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    work_tokens: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    evaluator_spend_micros: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    evaluator_tokens: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)


class MiraMethod(Model):
    release: Literal["research.mira@1"] = "research.mira@1"
    overrides: MiraOverrides | None = None


class ScaleMasParameters(Model):
    peers: StrictInt = Field(ge=1, le=32)
    maximum_tasks: StrictInt = Field(ge=1, le=1000)
    work_spend_micros: StrictInt = Field(ge=1, le=2**64 - 1)
    work_tokens: StrictInt = Field(ge=1, le=2**64 - 1)
    evaluator_spend_micros: StrictInt = Field(ge=1, le=2**64 - 1)
    evaluator_tokens: StrictInt = Field(ge=1, le=2**64 - 1)
    record_bytes: StrictInt = Field(ge=256, le=16384)
    retrieval_top_k: StrictInt = Field(ge=1, le=64)
    retrieval_bytes: StrictInt = Field(ge=4352, le=65536)
    retrieval_tokens: StrictInt = Field(ge=4352, le=65536)
    citation_weight: StrictInt = Field(ge=0, le=1000000)
    term_weight: StrictInt = Field(ge=0, le=1000000)
    decay_millis: StrictInt = Field(ge=1, le=2**63 - 1)

    @model_validator(mode="after")
    def validate_retrieval(self) -> Self:
        if (
            self.maximum_tasks < self.peers
            or self.retrieval_bytes < self.record_bytes + 4096
            or self.retrieval_tokens < self.retrieval_bytes
            or self.retrieval_tokens > self.work_tokens
        ):
            raise ValueError("ScaleMAS task and retrieval budgets are inconsistent")
        return self


class ScaleMasOverrides(Model):
    peers: StrictInt | None = Field(default=None, ge=1, le=32)
    maximum_tasks: StrictInt | None = Field(default=None, ge=1, le=1000)
    work_spend_micros: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    work_tokens: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    evaluator_spend_micros: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    evaluator_tokens: StrictInt | None = Field(default=None, ge=1, le=2**64 - 1)
    record_bytes: StrictInt | None = Field(default=None, ge=256, le=16384)
    retrieval_top_k: StrictInt | None = Field(default=None, ge=1, le=64)
    retrieval_bytes: StrictInt | None = Field(default=None, ge=4352, le=65536)
    retrieval_tokens: StrictInt | None = Field(default=None, ge=4352, le=65536)
    citation_weight: StrictInt | None = Field(default=None, ge=0, le=1000000)
    term_weight: StrictInt | None = Field(default=None, ge=0, le=1000000)
    decay_millis: StrictInt | None = Field(default=None, ge=1, le=2**63 - 1)


class ScaleMasMethod(Model):
    release: Literal["research.scalemas@1"] = "research.scalemas@1"
    overrides: ScaleMasOverrides | None = None


ResearchMethod = Annotated[
    ForkFlushMethod | ForgettingMethod | MiraMethod | ScaleMasMethod,
    Field(discriminator="release"),
]


class ResolvedResearchMethod(ResponseBase):
    release: Literal[
        "research.fork_flush@1", "research.forgetting@1",
        "research.mira@1", "research.scalemas@1",
    ]
    program_id: Literal[
        "research.fork_flush", "research.forgetting", "research.mira", "research.scalemas",
    ]
    program_version: Literal[1]
    definition_digest: Digest
    configuration_digest: Digest
    forge_profile_digest: Digest
    context_policy_digest: Digest
    parameters: ForkFlushParameters | ForgettingParameters | MiraParameters | ScaleMasParameters

    @model_validator(mode="after")
    def validate_pairing(self) -> Self:
        expected = {
            "research.fork_flush@1": ("research.fork_flush", ForkFlushParameters),
            "research.forgetting@1": ("research.forgetting", ForgettingParameters),
            "research.mira@1": ("research.mira", MiraParameters),
            "research.scalemas@1": ("research.scalemas", ScaleMasParameters),
        }[self.release]
        if self.program_id != expected[0] or not isinstance(self.parameters, expected[1]):
            raise ValueError("Research release and native program pairing differ")
        return self

    def satisfies(self, selection: ResearchMethod) -> bool:
        overrides = getattr(selection, "overrides", None)
        return self.release == selection.release and (
            overrides is None
            or all(
                getattr(self.parameters, name, None) == getattr(overrides, name)
                for name in overrides.model_fields_set
                if getattr(overrides, name) is not None
            )
        )
