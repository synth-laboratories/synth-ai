"""The finished run, its outputs and its final answer in one typed record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ._generated.models import RunCharge, RunOutput, RunSnapshot


class ObjectiveAnswerSource(BaseModel):
    """One record the answer cites (for example an uploaded artifact)."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    kind: str | None = None
    authority: str | None = None
    record_id: str | None = None
    revision: str | None = None
    digest_sha256: str | None = None


class ObjectiveAnswerBody(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    text: str
    by: str | None = None
    revision: int | None = None
    sources: tuple[ObjectiveAnswerSource, ...] = ()


class ObjectiveAnswer(BaseModel):
    """A `synth.objective-answer.v1` output: one objective's recorded answer."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    schema_version: Literal["synth.objective-answer.v1"]
    objective_id: str
    definition_revision: int | None = None
    answer: ObjectiveAnswerBody


@dataclass(frozen=True)
class RunResult:
    """What `swarms.result()` read for one finished run.

    `answer_output` is the output the answer came from: the most recently
    published available `text/*` output or, without one, the most recent
    available `application/json` objective answer (`synth.objective-answer.v1`).
    `content` is that output's digest-verified UTF-8 text (the raw JSON for an
    objective answer), `answer` is the answer text (the text output itself, or
    the objective answer's `answer.text`), and `objective_answer` is the parsed
    objective answer when that is the source. All are `None` when the run
    published neither. `outputs` lists every output in publication order.
    `charge` is the run's charge from the snapshot (`None` if Core omitted it).
    """

    run: RunSnapshot
    outputs: tuple[RunOutput, ...]
    answer_output: RunOutput | None
    content: str | None
    answer: str | None
    objective_answer: ObjectiveAnswer | None = None

    @property
    def charge(self) -> RunCharge | None:
        """The run's charge (`amount`, `final`) from the snapshot, when Core reported it."""
        return self.run.charge


def _media(output: RunOutput) -> str:
    reference = output.reference or {}
    return reference.get("media_type", "").split(";", 1)[0].strip().lower()


def _pinned(output: RunOutput) -> bool:
    reference = output.reference or {}
    return (
        output.state == "available"
        and reference.get("revision_id", "") != ""
        and reference.get("digest_sha256", "") != ""
    )


def answer_output(outputs: tuple[RunOutput, ...]) -> RunOutput | None:
    """The last available text output carrying a pinned revision and digest."""
    for output in reversed(outputs):
        if _pinned(output) and _media(output).startswith("text/"):
            return output
    return None


def json_answer_candidates(outputs: tuple[RunOutput, ...]) -> tuple[RunOutput, ...]:
    """Available pinned `application/json` outputs, most recent first."""
    return tuple(o for o in reversed(outputs) if _pinned(o) and _media(o) == "application/json")


def parse_objective_answer(content: str) -> ObjectiveAnswer | None:
    """The objective answer `content` carries, or `None` for any other JSON."""
    try:
        return ObjectiveAnswer.model_validate_json(content)
    except ValidationError:
        return None
