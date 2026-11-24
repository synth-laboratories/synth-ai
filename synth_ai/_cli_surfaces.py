"""Resource CLI commands over public SDK methods; see docs/cli.md."""

import argparse
import asyncio
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, RootModel

from ._generated.models import (
    AutoReloadRequest,
    CheckoutRequest,
    MintApiKeyRequest,
    RunLabelsRequest,
    SetSpendAlertsRequest,
    SubscriptionChangeRequest,
    UpdateApiKeyRequest,
    UploadCreateRequest,
)
from .aclient import AsyncSynthClient
from .research import ArtifactKind, ArtifactOrigin, ArtifactShareScope, AuthorKind, LogCategory

__all__ = ["configure", "execute_resource"]


def configure(
    parsers: Mapping[str, argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    groups = {
        "files": ("list", "get", "upload-text"),
        "uploads": ("prepare", "get", "finalize"),
        "keys": ("list", "mint", "update", "rotate", "revoke"),
        "organization": (
            "get",
            "usage-summary",
            "usage-records",
            "usage-export",
            "usage-export-status",
            "limits",
            "accept-terms",
        ),
        "billing": (
            "balance",
            "plans",
            "history",
            "statements",
            "checkout",
            "checkout-status",
            "subscription",
            "cancel-subscription",
            "resume-subscription",
            "preview-plan",
            "change-plan",
            "payment-method",
            "auto-reload",
            "set-auto-reload",
            "spend-alerts",
            "set-spend-alerts",
        ),
    }
    for group, actions in groups.items():
        children = parsers[group].add_subparsers(
            dest="resource_action", required=group != "billing"
        )
        for action in actions:
            child = children.add_parser(action, parents=[common])
            child.add_argument("--idempotency-key")
            if action in {"list", "history", "statements", "usage-records"}:
                child.add_argument("--limit", type=int, default=20)
                child.add_argument("--cursor")
            if group == "keys" and action == "list":
                child.add_argument("--purpose", choices=("personal", "automation"))
            if action in {"usage-summary", "usage-records", "usage-export", "history"}:
                child.add_argument("--since", required=action == "usage-export")
                child.add_argument("--until", required=action == "usage-export")
            if action in {"usage-records", "history"}:
                child.add_argument("--run-id")
            if action == "history":
                child.add_argument("--cause")
            if action == "statements":
                child.add_argument("--months", type=int)
            if action in {
                "prepare",
                "mint",
                "update",
                "checkout",
                "change-plan",
                "set-auto-reload",
                "set-spend-alerts",
            }:
                child.add_argument("--body", required=True, metavar="PATH")
            if (
                action
                in {
                    "get",
                    "finalize",
                    "update",
                    "rotate",
                    "revoke",
                    "checkout-status",
                    "usage-export-status",
                }
                and group != "organization"
            ):
                child.add_argument("identifier")
            if action == "usage-export-status":
                child.add_argument("identifier")
            if action == "preview-plan":
                child.add_argument("plan_id")
            if action == "upload-text":
                child.add_argument("text")
                child.add_argument("--name", required=True)
                child.add_argument("--media-type", default="text/plain")
            if action == "accept-terms":
                child.add_argument("--version", required=True)
                child.add_argument("--accepted-at", required=True)
    labels = parsers["labels"]
    labels.add_argument("run_id")
    labels.add_argument("--body", required=True, metavar="PATH")
    content = parsers["content"]
    content.add_argument("run_id")
    content.add_argument("output_id")
    content.add_argument("--out", required=True)
    content.add_argument("--revision", required=True)
    content.add_argument("--sha256", required=True)
    inspect = parsers["inspect"]
    inspect.add_argument("run_id")
    inspect.add_argument(
        "--view",
        choices=("program", "configuration", "sandboxes", "output", "wait-update", "review"),
        required=True,
    )
    inspect.add_argument("--output-id")
    inspect.add_argument("--timeout-seconds", type=int, default=30)
    parsers["families"].set_defaults(resource_action="list")

    art = parsers["artifacts"].add_subparsers(dest="resource_action", required=True)
    art_list = art.add_parser("list", parents=[common])
    art_list.add_argument("--notebook-id", required=True)
    art_list.add_argument("--run-id")
    art_get = art.add_parser("get", parents=[common])
    art_get.add_argument("artifact_id")
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
    art_revoke = art.add_parser("revoke-share", parents=[common])
    art_revoke.add_argument("artifact_id")

    exp = parsers["experiments"].add_subparsers(dest="resource_action", required=True)
    exp_list = exp.add_parser("list", parents=[common])
    exp_list.add_argument("--notebook-id", required=True)
    exp_list.add_argument("--run-id")
    exp_get = exp.add_parser("get", parents=[common])
    exp_get.add_argument("experiment_id")
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
    log_search.add_argument("--category", choices=[c.value for c in LogCategory])
    log_search.add_argument("--run-id")
    log_export = log.add_parser("export", parents=[common])
    log_export.add_argument("--notebook-id", required=True)
    log_export.add_argument("--run-id")
    log_export.add_argument("--format", choices=("markdown", "json"), default="markdown")


def _read_body_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _write_output_bytes(path: str, content: bytes) -> str:
    destination = Path(path)
    destination.write_bytes(content)
    return str(destination)


async def execute_resource(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel | None:
    match args.command:
        case "labels":
            labels = RunLabelsRequest.model_validate_json(
                await asyncio.to_thread(_read_body_bytes, args.body)
            )
            return await client.swarms.update_labels(
                args.run_id, labels.labels, expected_revision=labels.expected_revision
            )
        case "content":
            content = await client.swarms.output_content(
                args.run_id, args.output_id, revision_id=args.revision, digest_sha256=args.sha256
            )
            written = await asyncio.to_thread(_write_output_bytes, args.out, content)
            return RootModel[str](written)
        case "families":
            return await client.swarms.families()
        case "inspect":
            match args.view:
                case "program":
                    return await client.swarms.program(args.run_id)
                case "configuration":
                    return await client.swarms.configuration(args.run_id)
                case "sandboxes":
                    return await client.swarms.sandboxes(args.run_id)
                case "wait-update":
                    return await client.swarms.wait_update(
                        args.run_id, timeout_seconds=args.timeout_seconds
                    )
                case "output":
                    if args.output_id is None:
                        raise ValueError("output view requires --output-id")
                    return await client.swarms.output(args.run_id, args.output_id)
                case "review":
                    return await client.swarms.review(args.run_id)
                case _:
                    raise ValueError("unknown run view")
        case "files":
            match args.resource_action:
                case "list":
                    return await client.swarms.files(cursor=args.cursor, limit=args.limit)
                case "get":
                    return await client.swarms.file(args.identifier)
                case "upload-text":
                    return await client.swarms.upload_text(
                        args.text,
                        name=args.name,
                        media_type=args.media_type,
                        idempotency_key=args.idempotency_key,
                    )
                case _:
                    raise ValueError("unknown files action")
        case "uploads":
            match args.resource_action:
                case "get":
                    return await client.swarms.upload(args.identifier)
                case "finalize":
                    return await client.swarms.finalize_upload(
                        args.identifier, idempotency_key=args.idempotency_key
                    )
                case "prepare":
                    body = UploadCreateRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.swarms.prepare_upload(
                        name=body.name,
                        media_type=body.media_type,
                        size_bytes=body.size_bytes,
                        digest_sha256=body.digest_sha256,
                        idempotency_key=args.idempotency_key,
                    )
                case _:
                    raise ValueError("unknown uploads action")
        case "keys":
            match args.resource_action:
                case "list":
                    return await client.account.api_keys.list(
                        purpose=args.purpose, cursor=args.cursor, limit=args.limit
                    )
                case "rotate":
                    return await client.account.api_keys.rotate(
                        args.identifier, idempotency_key=args.idempotency_key
                    )
                case "revoke":
                    return await client.account.api_keys.revoke(args.identifier)
                case "mint":
                    key = MintApiKeyRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.account.api_keys.mint(
                        purpose=key.purpose,
                        name=key.name,
                        scopes=key.scopes,
                        expires_at=key.expires_at,
                        request_id=key.request_id,
                        idempotency_key=args.idempotency_key,
                    )
                case "update":
                    grants = UpdateApiKeyRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.account.api_keys.update_grants(
                        args.identifier,
                        scopes=grants.scopes,
                        expected_version=grants.expected_version,
                        name=grants.name,
                    )
                case _:
                    raise ValueError("unknown keys action")
        case "organization":
            match args.resource_action:
                case "get":
                    return await client.account.get()
                case "limits":
                    return await client.account.limits()
                case "usage-summary":
                    return await client.account.usage_summary(since=args.since, until=args.until)
                case "usage-records":
                    return await client.account.usage_records(
                        since=args.since,
                        until=args.until,
                        run_id=args.run_id,
                        cursor=args.cursor,
                        limit=args.limit,
                    )
                case "usage-export":
                    return await client.account.export_usage(
                        since=args.since, until=args.until, idempotency_key=args.idempotency_key
                    )
                case "usage-export-status":
                    return await client.account.usage_export(args.identifier)
                case "accept-terms":
                    return await client.account.accept_terms(
                        version=args.version,
                        accepted_at=args.accepted_at,
                        idempotency_key=args.idempotency_key,
                    )
                case _:
                    raise ValueError("unknown account action")
        case "billing":
            match args.resource_action:
                case None | "balance":
                    return await client.billing.balance()
                case "plans":
                    return await client.billing.plans()
                case "history":
                    return await client.billing.history(
                        cursor=args.cursor,
                        limit=args.limit,
                        cause=args.cause,
                        run_id=args.run_id,
                        since=args.since,
                        until=args.until,
                    )
                case "statements":
                    return await client.billing.statements(
                        months=args.months, cursor=args.cursor, limit=args.limit
                    )
                case "checkout-status":
                    return await client.billing.checkout_status(args.identifier)
                case "subscription":
                    return await client.billing.subscription()
                case "cancel-subscription":
                    return await client.billing.cancel_subscription(
                        idempotency_key=args.idempotency_key
                    )
                case "resume-subscription":
                    return await client.billing.resume_subscription(
                        idempotency_key=args.idempotency_key
                    )
                case "preview-plan":
                    return await client.billing.preview_subscription_change(args.plan_id)
                case "payment-method":
                    return await client.billing.payment_method_session(
                        idempotency_key=args.idempotency_key
                    )
                case "auto-reload":
                    return await client.billing.auto_reload()
                case "spend-alerts":
                    return await client.billing.spend_alerts()
                case "checkout":
                    checkout = CheckoutRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.billing.checkout(
                        plan_id=checkout.plan_id,
                        pack_id=checkout.pack_id,
                        credit=checkout.credit,
                        idempotency_key=args.idempotency_key,
                    )
                case "change-plan":
                    plan = SubscriptionChangeRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.billing.change_subscription_plan(
                        plan.plan_id, idempotency_key=args.idempotency_key
                    )
                case "set-auto-reload":
                    reload = AutoReloadRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.billing.set_auto_reload(
                        enabled=reload.enabled,
                        threshold=reload.threshold,
                        amount=reload.amount,
                        disclosure_version=reload.disclosure_version,
                        idempotency_key=args.idempotency_key,
                    )
                case "set-spend-alerts":
                    alerts = SetSpendAlertsRequest.model_validate_json(
                        await asyncio.to_thread(_read_body_bytes, args.body)
                    )
                    return await client.billing.set_spend_alerts_request(
                        alerts, idempotency_key=args.idempotency_key
                    )
                case _:
                    raise ValueError("unknown billing action")
        case "artifacts":
            match args.resource_action:
                case "list":
                    return await client.artifacts.list_artifacts(
                        args.notebook_id, run_id=args.run_id
                    )
                case "get":
                    return await client.artifacts.get_artifact(args.artifact_id)
                case "content":
                    body = await client.artifacts.get_artifact_content(
                        args.artifact_id,
                        revision_id=args.revision,
                        digest_sha256=args.sha256,
                    )
                    if args.out is not None:
                        written = await asyncio.to_thread(
                            _write_output_bytes, args.out, body.encode("utf-8")
                        )
                        return RootModel[str](written)
                    return RootModel[str](body)
                case "create":
                    content = (await asyncio.to_thread(_read_body_bytes, args.file)).decode("utf-8")
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
                    content = (await asyncio.to_thread(_read_body_bytes, args.file)).decode("utf-8")
                    return await client.artifacts.update_artifact(
                        args.artifact_id,
                        content,
                        args.summary,
                        idempotency_key=args.idempotency_key,
                    )
                case "share":
                    return await client.artifacts.share_artifact(
                        args.artifact_id,
                        scope=args.scope,
                        idempotency_key=args.idempotency_key,
                    )
                case "revoke-share":
                    return await client.artifacts.revoke_artifact_share(args.artifact_id)
                case _:
                    raise ValueError("unknown artifacts action")
        case "experiments":
            match args.resource_action:
                case "list":
                    return await client.experiments.list_experiments(
                        args.notebook_id, run_id=args.run_id
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
        case "research-log":
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
                    )
                case "export":
                    return await client.research_logs.export_log(
                        args.notebook_id, run_id=args.run_id, fmt=args.format
                    )
                case _:
                    raise ValueError("unknown research-log action")
        case _:
            raise ValueError("unknown resource command")
