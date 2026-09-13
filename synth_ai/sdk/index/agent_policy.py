"""Opt-in Index access for a Swarm run; mirror of backend ``IndexAccessPolicy``.

See sibling docs/drafts/synth-index-agent-integration-2026-09-12.md §4. Sent as
``RunPolicy.index``. Absent means disabled; enabling Index never enables charges.
Private access needs selected collections and an explicit spend cap.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from .contracts import Identifier, IndexContract


class IndexAccessPolicy(IndexContract):
    enabled: StrictBool = False
    visibility: Literal["public", "private"] = "public"
    collection_ids: tuple[Identifier, ...] = Field(default=(), max_length=16)
    max_searches: Annotated[StrictInt, Field(ge=0, le=100)] = 10
    max_contents: Annotated[StrictInt, Field(ge=0, le=200)] = 20
    max_context_tokens: Annotated[StrictInt, Field(ge=0, le=64_000)] = 8_000
    private_spend_cap_cents: Annotated[StrictInt, Field(ge=1, le=100_000)] | None = None

    @model_validator(mode="after")
    def check_audience(self) -> Self:
        if self.visibility == "private":
            if not self.collection_ids or self.private_spend_cap_cents is None:
                raise ValueError(
                    "private Index access requires selected collections and a spend cap"
                )
        elif self.collection_ids or self.private_spend_cap_cents is not None:
            raise ValueError("public Index access cannot carry private scope or spend")
        return self
