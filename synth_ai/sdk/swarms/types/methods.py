"""Fork-and-Flush selection. See docs/swarms.md.

These contracts express intent, not deployment capability. Omission preserves
server defaults; unsupported releases must refuse before any paid execution.
"""

from __future__ import annotations

from typing import Literal, Self

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
            raise ValueError("Fork rounds must leave positive exploitation intervals within budget")
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
    rounds: tuple[ForkFlushRound, ...] | None = Field(default=None, min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if self.rounds is not None:
            validate_rounds(self.rounds)
        return self


class ForkFlushMethod(Model):
    release: Literal["research.fork_flush@1"] = "research.fork_flush@1"
    overrides: ForkFlushOverrides | None = None


class ResolvedResearchMethod(ResponseBase):
    release: Literal["research.fork_flush@1"]
    program_id: Literal["research.fork_flush"]
    program_version: Literal[1]
    definition_digest: Digest
    configuration_digest: Digest
    forge_profile_digest: Digest
    context_policy_digest: Digest
    parameters: ForkFlushParameters

    def satisfies(self, selection: ForkFlushMethod) -> bool:
        overrides = selection.overrides
        return self.release == selection.release and (
            overrides is None
            or (
                (overrides.branches is None or self.parameters.branches == overrides.branches)
                and (overrides.rounds is None or self.parameters.rounds == overrides.rounds)
            )
        )
