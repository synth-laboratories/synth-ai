"""Private source facts, never Index release decisions.

See notes/specifications/synth-index/forge-source-notices.md.
"""

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from synth_ai.sdk.research.contracts.forge.contracts import ExactReference

from .contracts import ContributionReference, Identifier, IndexContract
from .contributions import VersionedForgeSource


class ForgeSourceNoticeItem(IndexContract):
    cursor: int = Field(ge=1, strict=True)
    notice: ExactReference
    operation_id: Identifier
    recorded_at: AwareDatetime
    original: ExactReference
    replacement: ExactReference | None = None
    disposition: Literal["correction", "supersession", "retraction"]
    reason: str = Field(min_length=1, max_length=8000)


class ForgeSourceNoticePage(IndexContract):
    schema_version: Literal["synth.index.forge-source-notices.v1"] = (
        "synth.index.forge-source-notices.v1"
    )
    source: VersionedForgeSource
    reference: ContributionReference
    release: ExactReference
    items: tuple[ForgeSourceNoticeItem, ...] = Field(max_length=100)
    # A checkpoint in the scoped Forge stream, including unrelated events.
    next_cursor: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def exact_bindings(self):
        if (
            self.release.authority != "index"
            or self.release.kind != "contribution"
            or self.release.record_id != self.reference.contribution_id
            or self.release.revision != self.reference.revision_id
        ):
            raise ValueError("notice page requires an exact Index revision")
        cursors = tuple(item.cursor for item in self.items)
        if cursors != tuple(sorted(set(cursors))) or any(
            item.cursor > self.next_cursor
            or item.notice.authority != "forge"
            or item.notice.kind != "source_notice"
            or item.original.authority != "forge"
            or (item.disposition != "retraction" and item.replacement is None)
            or (
                item.replacement is not None
                and (
                    item.replacement.authority != "forge"
                    or item.replacement.kind != item.original.kind
                    or item.replacement == item.original
                )
            )
            for item in self.items
        ):
            raise ValueError("notice page contains invalid exact facts or cursors")
        return self
