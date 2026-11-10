"""The finished run, its outputs and its final text answer in one typed record."""

from __future__ import annotations

from dataclasses import dataclass

from ._generated.models import RunOutput, RunSnapshot


@dataclass(frozen=True)
class RunResult:
    """What `swarms.result()` read for one finished run.

    `answer` is the most recently published available output whose recorded media
    type is `text/*`; `content` is its digest-verified UTF-8 text. Both are `None`
    when the run published no text output. `outputs` lists every output in the
    server's publication order.
    """

    run: RunSnapshot
    outputs: tuple[RunOutput, ...]
    answer: RunOutput | None
    content: str | None


def answer_output(outputs: tuple[RunOutput, ...]) -> RunOutput | None:
    """The last available text output carrying a pinned revision and digest."""
    for output in reversed(outputs):
        reference = output.reference or {}
        media = reference.get("media_type", "").split(";", 1)[0].strip().lower()
        if (
            output.state == "available"
            and media.startswith("text/")
            and reference.get("revision_id", "") != ""
            and reference.get("digest_sha256", "") != ""
        ):
            return output
    return None
