"""Bounded public catalogue cards; notes/specifications/synth-index/public-catalogue.md."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictInt

from .contracts import (
    ContributionKind,
    ContributionReference,
    Identifier,
    IndexContract,
    ResearchArea,
    ShortText,
    WorkflowStage,
)
from .lifecycle import Citation, Title


class PublicContributionCard(IndexContract):
    """Only metadata from the exact current, approved public publication."""

    reference: ContributionReference
    title: Title
    abstract: ShortText
    kind: ContributionKind
    research_areas: tuple[ResearchArea, ...] = Field(max_length=11)
    workflow_stages: tuple[WorkflowStage, ...] = Field(max_length=6)
    tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    contributor_ids: tuple[Identifier, ...] = Field(max_length=32)
    published_at: AwareDatetime | None = None
    reproduction_level: Literal["inspectable", "runnable", "reproduced", "not_applicable"]
    observed: ShortText
    limitations: ShortText
    citation: Citation


class PublicCatalogue(IndexContract):
    """One bounded page and the count under exactly the same public filters."""

    items: tuple[PublicContributionCard, ...] = Field(max_length=32)
    next_cursor: Identifier | None = None
    total: Annotated[StrictInt, Field(ge=0)]
