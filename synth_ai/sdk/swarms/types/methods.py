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


# These names reserve public intent only. They remain unavailable until a
# published runtime/profile/release pairing exists; no synthetic pins are made.
class MiraMethod(Model):
    release: Literal["research.mira@1"] = "research.mira@1"


class ScaleMasMethod(Model):
    release: Literal["research.scalemas@1"] = "research.scalemas@1"


ResearchMethod = Annotated[
    ForkFlushMethod | ForgettingMethod | MiraMethod | ScaleMasMethod,
    Field(discriminator="release"),
]


class ResolvedResearchMethod(ResponseBase):
    release: Literal["research.fork_flush@1", "research.forgetting@1"]
    program_id: Literal["research.fork_flush", "research.forgetting"]
    program_version: Literal[1]
    definition_digest: Digest
    configuration_digest: Digest
    forge_profile_digest: Digest
    context_policy_digest: Digest
    parameters: ForkFlushParameters | ForgettingParameters

    @model_validator(mode="after")
    def validate_pairing(self) -> Self:
        expected = {
            "research.fork_flush@1": ("research.fork_flush", ForkFlushParameters),
            "research.forgetting@1": ("research.forgetting", ForgettingParameters),
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
