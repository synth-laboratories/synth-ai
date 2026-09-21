"""Typed grounded-answer contracts for Synth Index."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator

from .contracts import ContributionReference, Identifier, IndexContract, require_unique
from .search import (
    SearchBillingConstraints,
    SearchContent,
    SearchExecutionLimits,
    SearchExecutionVersions,
    SearchFilters,
    SearchMode,
    SearchScope,
)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AnswerSpec(IndexContract):
    query: Annotated[str, Field(min_length=1, max_length=8192)]
    mode: SearchMode = SearchMode.FAST
    scope: SearchScope = Field(default_factory=SearchScope)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    content: SearchContent = Field(
        default_factory=lambda: SearchContent(max_results=5, max_excerpts_per_result=2)
    )
    limits: SearchExecutionLimits | None = None
    billing: SearchBillingConstraints = Field(default_factory=SearchBillingConstraints)
    max_answer_tokens: Annotated[StrictInt, Field(ge=64, le=4096)] = 1024
    max_answer_cost_usd_micros: Annotated[StrictInt, Field(ge=1)] | None = None

    @field_validator("query")
    @classmethod
    def check_query_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 8192 or "\x00" in value:
            raise ValueError("query must fit 8192 UTF-8 bytes and contain no NUL")
        return value

    @model_validator(mode="after")
    def check_mode_options(self) -> Self:
        if self.mode is SearchMode.FAST and self.limits is not None:
            raise ValueError("execution limits are supported only for deep answers")
        return self


class AnswerCitation(IndexContract):
    citation_id: Annotated[str, Field(pattern=r"^c[1-9][0-9]{0,2}$")]
    reference: ContributionReference
    asset_id: Identifier
    chunk_id: Identifier
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    start_byte: Annotated[StrictInt, Field(ge=0)]
    end_byte: Annotated[StrictInt, Field(gt=0)]
    quote: Annotated[str, Field(min_length=1, max_length=2048)]

    @model_validator(mode="after")
    def check_exact_span(self) -> Self:
        if self.end_byte - self.start_byte != len(self.quote.encode("utf-8")):
            raise ValueError("answer citation quote must equal its exact byte span")
        return self


class AnswerClaim(IndexContract):
    text: Annotated[str, Field(min_length=1, max_length=2048)]
    citation_ids: tuple[Annotated[str, Field(pattern=r"^c[1-9][0-9]{0,2}$")], ...] = Field(
        min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def check_citations(self) -> Self:
        require_unique(self.citation_ids, "claim citation IDs")
        return self


class AnswerUsage(IndexContract):
    retrieval_input_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    retrieval_output_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    admission_input_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    admission_output_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    synthesis_input_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    synthesis_output_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    inference_cost_usd_micros: Annotated[StrictInt, Field(ge=0)] = 0


class AnswerResult(IndexContract):
    answer_id: Identifier
    search_id: Identifier
    request_id: Identifier
    query: str
    mode: SearchMode
    status: AnswerStatus
    answer: Annotated[str, Field(min_length=1, max_length=16_384)] | None = None
    insufficient_evidence_reason: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    claims: tuple[AnswerClaim, ...] = Field(default=(), max_length=16)
    citations: tuple[AnswerCitation, ...] = Field(default=(), max_length=20)
    retrieval_versions: SearchExecutionVersions
    admission_policy_version: Literal["synth.index.answer-admission.v1"]
    synthesis_policy_version: Literal["synth.index.cited-synthesis.v1"]
    model_version: (
        Annotated[
            str,
            Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./-]{0,255}$"),
        ]
        | None
    ) = None
    usage: AnswerUsage = Field(default_factory=AnswerUsage)

    @model_validator(mode="after")
    def check_grounding(self) -> Self:
        require_unique(
            tuple(citation.citation_id for citation in self.citations),
            "answer citation IDs",
        )
        known = {citation.citation_id for citation in self.citations}
        cited = {citation_id for claim in self.claims for citation_id in claim.citation_ids}
        if not cited.issubset(known):
            raise ValueError("answer claim cites evidence outside the delivered set")
        if self.status is AnswerStatus.ANSWERED:
            if not self.answer or not self.claims or not self.citations:
                raise ValueError("answered result requires answer, claims and citations")
            if self.answer != "\n\n".join(claim.text for claim in self.claims):
                raise ValueError("answer must be rendered exactly from its cited claims")
            if cited != known:
                raise ValueError("delivered answer citations must all support a claim")
            if self.insufficient_evidence_reason is not None or self.model_version is None:
                raise ValueError("answered result requires model identity and no refusal")
        elif (
            self.answer is not None
            or self.claims
            or self.citations
            or self.insufficient_evidence_reason is None
        ):
            raise ValueError("insufficient evidence result cannot contain an answer")
        return self
