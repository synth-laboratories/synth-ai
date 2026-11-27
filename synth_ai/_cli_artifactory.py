"""CLI configuration, execution and human views for `synth notebooks` and `synth artifactory`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ._cli_view import table
from ._validation import safe_validation_detail
from .artifactory import (
    ArtifactoryAuthorKind,
    ArtifactoryCategory,
    ArtifactoryFinding,
    ArtifactoryHypothesis,
    ArtifactoryOrigin,
    ArtifactoryPayload,
    ArtifactoryPreference,
    ArtifactoryRecord,
    ArtifactoryRecordList,
    NotebookPage,
)
from .errors import ArtifactoryPayloadError

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

    from pydantic import BaseModel

    from .aclient import AsyncSynthClient

__all__ = [
    "ARTIFACTORY_COMMANDS",
    "configure_artifactory",
    "execute_artifactory",
    "parse_payload",
    "render_artifactory",
]

ARTIFACTORY_COMMANDS = frozenset({"notebooks", "artifactory"})
_PAYLOAD_MODELS: dict[
    ArtifactoryCategory,
    type[ArtifactoryPreference] | type[ArtifactoryHypothesis] | type[ArtifactoryFinding],
] = {
    ArtifactoryCategory.PREFERENCE: ArtifactoryPreference,
    ArtifactoryCategory.HYPOTHESIS: ArtifactoryHypothesis,
    ArtifactoryCategory.FINDING: ArtifactoryFinding,
}


def configure_artifactory(
    parsers: Mapping[str, argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    """Register `notebooks list` and the `artifactory` record subcommands."""
    notebooks = parsers["notebooks"].add_subparsers(dest="resource_action", required=True)
    nb_list = notebooks.add_parser("list", parents=[common])
    nb_list.add_argument("--limit", type=int, default=20)
    nb_list.add_argument("--cursor")

    categories = [c.value for c in ArtifactoryCategory]
    origins = [o.value for o in ArtifactoryOrigin]
    authors = [a.value for a in ArtifactoryAuthorKind]
    art = parsers["artifactory"].add_subparsers(dest="resource_action", required=True)
    art_list = art.add_parser("list", parents=[common])
    art_list.add_argument("--notebook-id", required=True)
    art_list.add_argument("--category", choices=categories)
    art_list.add_argument("--origin", choices=origins)
    art_list.add_argument("--run-id")
    art_list.add_argument("--pinned", action="store_true", help="only pinned records")
    art.add_parser("get", parents=[common]).add_argument("record_id", metavar="RECORD")
    preference = art.add_parser("preference", parents=[common])
    preference.add_argument("--notebook-id", required=True)
    preference.add_argument("--run-id")
    create = art.add_parser("create", parents=[common])
    create.add_argument("--notebook-id", required=True)
    create.add_argument("--title", required=True)
    create.add_argument("--category", choices=categories, required=True)
    create.add_argument("--file", required=True, metavar="PATH", help="JSON payload for --category")
    create.add_argument(
        "--origin",
        choices=origins,
        required=True,
        help="human needs a Console session or personal API key",
    )
    create.add_argument("--run-id")
    create.add_argument("--author-kind", choices=authors)
    create.add_argument("--summary")
    create.add_argument("--pin", action="store_true")
    create.add_argument("--idempotency-key")
    revise = art.add_parser("revise", parents=[common])
    revise.add_argument("record_id", metavar="RECORD")
    revise.add_argument("--category", choices=categories, help="schema for --file")
    revise.add_argument("--file", metavar="PATH", help="JSON payload; requires --category")
    revise.add_argument("--title")
    revise.add_argument("--summary")
    pin = revise.add_mutually_exclusive_group()
    pin.add_argument("--pin", dest="pinned", action="store_true", default=None)
    pin.add_argument("--unpin", dest="pinned", action="store_false")
    revise.add_argument("--author-kind", choices=authors)
    revise.add_argument("--expected-revision", type=int)
    revise.add_argument("--expected-digest")
    revise.add_argument("--idempotency-key")


def _read_text(path: str) -> str:
    try:
        return Path(path).read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactoryPayloadError(
            f"--file unreadable: {path}: {type(error).__name__}"
        ) from None


def parse_payload(category: ArtifactoryCategory | str, raw: str) -> ArtifactoryPayload:
    """Validate a JSON payload against the schema for its record category."""
    model = _PAYLOAD_MODELS[ArtifactoryCategory(category)]
    try:
        return model.model_validate_json(raw)
    except ValidationError as error:
        raise ArtifactoryPayloadError(
            f"--file invalid {model.__name__}: {safe_validation_detail(error)}"
        ) from None


async def _load_payload(category: str | None, path: str | None) -> ArtifactoryPayload | None:
    if path is None:
        if category is not None:
            raise ArtifactoryPayloadError("--category requires --file")
        return None
    if category is None:
        raise ArtifactoryPayloadError("--file requires --category")
    return parse_payload(category, await asyncio.to_thread(_read_text, path))


async def execute_artifactory(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
    """Execute one `notebooks` or `artifactory` CLI subcommand."""
    if args.command == "notebooks":
        return await client.notebooks.list_notebooks(limit=args.limit, cursor=args.cursor)
    match args.resource_action:
        case "list":
            return await client.artifactory.list_records(
                args.notebook_id,
                category=args.category,
                origin=args.origin,
                run_id=args.run_id,
                pinned_only=args.pinned,
            )
        case "get":
            return await client.artifactory.get_record(args.record_id)
        case "preference":
            return await client.artifactory.resolve_preference(args.notebook_id, run_id=args.run_id)
        case "create":
            payload = parse_payload(args.category, await asyncio.to_thread(_read_text, args.file))
            return await client.artifactory.create_record(
                args.notebook_id,
                args.title,
                payload,
                origin=args.origin,
                run_id=args.run_id,
                author_kind=args.author_kind,
                pinned=True if args.pin else None,
                summary=args.summary,
                idempotency_key=args.idempotency_key,
            )
        case "revise":
            return await client.artifactory.revise_record(
                args.record_id,
                payload=await _load_payload(args.category, args.file),
                title=args.title,
                summary=args.summary,
                pinned=args.pinned,
                author_kind=args.author_kind,
                expected_revision=args.expected_revision,
                expected_supersedes_digest=args.expected_digest,
                idempotency_key=args.idempotency_key,
            )
        case _:
            raise ValueError("unknown artifactory action")


def _headline(record: ArtifactoryRecord) -> str:
    if record.preference is not None:
        objectives = ", ".join(
            f"{o.direction.value} {o.sensor_id}" for o in record.preference.objectives
        )
        return f"{record.preference.aggregation_mode.value}: {objectives or '-'}"
    if record.hypothesis is not None:
        return record.hypothesis.statement
    if record.finding is not None:
        dead_end = " (dead end)" if record.finding.dead_end else ""
        return record.finding.summary + dead_end
    return "-"


def _record(record: ArtifactoryRecord) -> str:
    rows: list[tuple[str, object]] = [
        ("record", record.record_id),
        ("title", record.title),
        ("category", record.category.value),
        ("origin", record.origin.value),
        ("pinned", "yes" if record.pinned else "no"),
        ("notebook", record.notebook_id),
        ("run", record.run_id),
        ("revision", f"{record.current_revision} ({record.current_digest_sha256[:12]})"),
        ("schema", record.schema_version),
        ("content", _headline(record)),
        ("updated", record.updated_at),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [label.ljust(width) + "  " + ("-" if v is None else str(v)) for label, v in rows]
    history = table(
        ("REV", "AUTHOR", "PINNED", "CREATED", "SUMMARY"),
        [
            (r.revision, r.author_kind.value, "yes" if r.pinned else "no", r.created_at, r.summary)
            for r in record.revisions
        ],
    )
    return "\n".join(lines) + "\n\n" + history


def render_artifactory(value: BaseModel) -> str | None:
    """Format a notebooks/artifactory response for humans, or None to fall back to JSON."""
    if isinstance(value, NotebookPage):
        body = table(
            ("NOTEBOOK", "ARTIFACTORY", "ARTIFACTS", "EXPERIMENTS", "LOG", "UPDATED"),
            [
                (
                    n.notebook_id,
                    n.artifactory_count,
                    n.artifact_count,
                    n.experiment_count,
                    n.log_entry_count if n.has_research_log else None,
                    n.updated_at,
                )
                for n in value.items
            ],
        )
        if value.has_more and value.next_cursor is not None:
            body += f"\nmore: --cursor {value.next_cursor}"
        return body
    if isinstance(value, ArtifactoryRecordList):
        if not value.items:
            return "no records"
        return table(
            ("RECORD", "CATEGORY", "ORIGIN", "PINNED", "REV", "TITLE"),
            [
                (
                    r.record_id,
                    r.category.value,
                    r.origin.value,
                    "yes" if r.pinned else "no",
                    r.current_revision,
                    r.title,
                )
                for r in value.items
            ],
        )
    if isinstance(value, ArtifactoryRecord):
        return _record(value)
    return None
