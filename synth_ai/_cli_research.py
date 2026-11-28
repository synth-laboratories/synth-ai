"""CLI configuration and execution for artifacts, experiments, and research logs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, RootModel

from .research import ArtifactKind, ArtifactOrigin, ArtifactShareScope, AuthorKind, LogCategory

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

    from .aclient import AsyncSynthClient

__all__ = ["configure_research", "execute_research"]


def read_utf8_file(path: str) -> str:
    """Read a UTF-8 input file with clear, bounded error reporting."""
    try:
        return Path(path).read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"--file unreadable: {path}: {type(error).__name__}") from error


def write_output_bytes(path: str, content: bytes) -> str:
    """Write output bytes to a local file path and return its string path."""
    destination = Path(path)
    destination.write_bytes(content)
    return str(destination)


def add_page_arguments(parser: argparse.ArgumentParser) -> None:
    """Register `--limit` (1-100; server default 20) and `--cursor` on a list subcommand."""
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cursor")


def configure_research(
    parsers: Mapping[str, argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    """Register CLI subcommands for artifacts, experiments, and research-log."""
    art = parsers["artifacts"].add_subparsers(dest="resource_action", required=True)
    art_list = art.add_parser("list", parents=[common])
    art_list.add_argument("--notebook-id", required=True)
    art_list.add_argument("--run-id")
    art_list.add_argument("--kind", choices=[k.value for k in ArtifactKind])
    art_list.add_argument("--origin", choices=[o.value for o in ArtifactOrigin])
    art_list.add_argument("--q", metavar="TEXT")
    add_page_arguments(art_list)
    art.add_parser("get", parents=[common]).add_argument("artifact_id")
    art_content = art.add_parser("content", parents=[common])
    art_content.add_argument("artifact_id")
    art_content.add_argument("--revision")
    art_content.add_argument("--sha256")
    art_content.add_argument("--out")
    art_create = art.add_parser("create", parents=[common])
    art_create.add_argument("--notebook-id", required=True)
    art_create.add_argument("--run-id")
    art_create.add_argument("--title", required=True)
    art_create.add_argument("--kind", choices=[k.value for k in ArtifactKind], required=True)
    art_create.add_argument("--origin", choices=[o.value for o in ArtifactOrigin], default="human")
    art_create.add_argument("--summary", required=True)
    art_create.add_argument("--file", required=True, metavar="PATH")
    art_create.add_argument("--idempotency-key")
    art_update = art.add_parser("update", parents=[common])
    art_update.add_argument("artifact_id")
    art_update.add_argument("--summary", required=True)
    art_update.add_argument("--file", required=True, metavar="PATH")
    art_update.add_argument("--idempotency-key")
    art_share = art.add_parser("share", parents=[common])
    art_share.add_argument("artifact_id")
    art_share.add_argument("--scope", choices=[s.value for s in ArtifactShareScope], default="org")
    art_share.add_argument("--idempotency-key")
    art.add_parser("revoke-share", parents=[common]).add_argument("artifact_id")

    exp = parsers["experiments"].add_subparsers(dest="resource_action", required=True)
    exp_list = exp.add_parser("list", parents=[common])
    exp_list.add_argument("--notebook-id", required=True)
    exp_list.add_argument("--run-id")
    exp_list.add_argument("--q", metavar="TEXT")
    add_page_arguments(exp_list)
    exp.add_parser("get", parents=[common]).add_argument("experiment_id")
    exp_create = exp.add_parser("create", parents=[common])
    exp_create.add_argument("--notebook-id", required=True)
    exp_create.add_argument("--run-id", required=True)
    exp_create.add_argument("--title", required=True)
    exp_create.add_argument("--specification", required=True)
    exp_create.add_argument("--hypothesis")
    exp_create.add_argument("--arm", action="append", required=True)
    exp_create.add_argument("--idempotency-key")
    exp_revise = exp.add_parser("revise", parents=[common])
    exp_revise.add_argument("experiment_id")
    exp_revise.add_argument("--title")
    exp_revise.add_argument("--specification")
    exp_revise.add_argument("--hypothesis")
    exp_revise.add_argument("--arm", action="append")
    exp_revise.add_argument("--stopping-criteria")
    exp_revise.add_argument("--idempotency-key")

    log = parsers["research-log"].add_subparsers(dest="resource_action", required=True)
    log_init = log.add_parser("init", parents=[common])
    log_init.add_argument("--notebook-id", required=True)
    log_init.add_argument("--title", required=True)
    log_init.add_argument("--summary")
    log_init.add_argument("--idempotency-key")
    log_append = log.add_parser("append", parents=[common])
    log_append.add_argument("--notebook-id", required=True)
    log_append.add_argument("--run-id")
    log_append.add_argument("--category", choices=[c.value for c in LogCategory], required=True)
    log_append.add_argument("--text", required=True)
    log_append.add_argument(
        "--author-kind", choices=[a.value for a in AuthorKind], default="local_agent"
    )
    log_append.add_argument("--author-label")
    log_append.add_argument("--idempotency-key")
    log_correct = log.add_parser("correct", parents=[common])
    log_correct.add_argument("entry_id")
    log_correct.add_argument("--notebook-id", required=True)
    log_correct.add_argument("--run-id")
    log_correct.add_argument("--text", required=True)
    log_correct.add_argument("--idempotency-key")
    log_search = log.add_parser("search", parents=[common])
    log_search.add_argument("--notebook-id", required=True)
    log_search.add_argument("--query")
    log_search.add_argument("--q", metavar="TEXT")
    log_search.add_argument("--category", choices=[c.value for c in LogCategory])
    log_search.add_argument("--run-id")
    log_search.add_argument("--author-kind", choices=[a.value for a in AuthorKind])
    add_page_arguments(log_search)
    log_export = log.add_parser("export", parents=[common])
    log_export.add_argument("--notebook-id", required=True)
    log_export.add_argument("--run-id")
    log_export.add_argument("--format", choices=("markdown", "json"), default="markdown")


async def _execute_artifacts(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
    match args.resource_action:
        case "list":
            return await client.artifacts.list_artifacts(
                args.notebook_id,
                run_id=args.run_id,
                kind=args.kind,
                origin=args.origin,
                q=args.q,
                limit=args.limit,
                cursor=args.cursor,
            )
        case "get":
            return await client.artifacts.get_artifact(args.artifact_id)
        case "content":
            body = await client.artifacts.get_artifact_content(
                args.artifact_id, revision_id=args.revision, digest_sha256=args.sha256
            )
            if args.out is not None:
                written = await asyncio.to_thread(
                    write_output_bytes, args.out, body.encode("utf-8")
                )
                return RootModel[str](written)
            return RootModel[str](body)
        case "create":
            content = await asyncio.to_thread(read_utf8_file, args.file)
            return await client.artifacts.create_artifact(
                args.notebook_id,
                args.title,
                args.kind,
                content,
                args.summary,
                origin=args.origin,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
            )
        case "update":
            content = await asyncio.to_thread(read_utf8_file, args.file)
            return await client.artifacts.update_artifact(
                args.artifact_id, content, args.summary, idempotency_key=args.idempotency_key
            )
        case "share":
            return await client.artifacts.share_artifact(
                args.artifact_id, scope=args.scope, idempotency_key=args.idempotency_key
            )
        case "revoke-share":
            return await client.artifacts.revoke_artifact_share(args.artifact_id)
        case _:
            raise ValueError("unknown artifacts action")


async def _execute_experiments(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
    match args.resource_action:
        case "list":
            return await client.experiments.list_experiments(
                args.notebook_id,
                run_id=args.run_id,
                q=args.q,
                limit=args.limit,
                cursor=args.cursor,
            )
        case "get":
            return await client.experiments.get_experiment(args.experiment_id)
        case "create":
            return await client.experiments.create_experiment(
                args.notebook_id,
                args.run_id,
                args.title,
                args.specification,
                args.arm,
                hypothesis=args.hypothesis,
                idempotency_key=args.idempotency_key,
            )
        case "revise":
            return await client.experiments.revise_experiment(
                args.experiment_id,
                title=args.title,
                specification=args.specification,
                hypothesis=args.hypothesis,
                arms=args.arm,
                stopping_criteria=args.stopping_criteria,
                idempotency_key=args.idempotency_key,
            )
        case _:
            raise ValueError("unknown experiments action")


async def _execute_research_log(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
    match args.resource_action:
        case "init":
            return await client.research_logs.create_log(
                args.notebook_id,
                title=args.title,
                summary=args.summary,
                idempotency_key=args.idempotency_key,
            )
        case "append":
            return await client.research_logs.append_entry(
                args.notebook_id,
                args.category,
                args.text,
                run_id=args.run_id,
                author_kind=args.author_kind,
                author_label=args.author_label,
                idempotency_key=args.idempotency_key,
            )
        case "correct":
            return await client.research_logs.correct_entry(
                args.notebook_id,
                args.entry_id,
                args.text,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
            )
        case "search":
            return await client.research_logs.search_entries(
                args.notebook_id,
                query=args.query,
                category=args.category,
                run_id=args.run_id,
                author_kind=args.author_kind,
                q=args.q,
                limit=args.limit,
                cursor=args.cursor,
            )
        case "export":
            return await client.research_logs.export_log(
                args.notebook_id, run_id=args.run_id, fmt=args.format
            )
        case _:
            raise ValueError("unknown research-log action")


async def execute_research(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
    """Execute one artifacts, experiments, or research-log CLI subcommand."""
    match args.command:
        case "artifacts":
            return await _execute_artifacts(client, args)
        case "experiments":
            return await _execute_experiments(client, args)
        case "research-log":
            return await _execute_research_log(client, args)
        case _:
            raise ValueError("unknown research command")
