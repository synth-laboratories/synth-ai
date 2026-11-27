"""Resource CLI commands over public SDK methods; see docs/cli.md."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, RootModel, ValidationError

from ._cli_research import configure_research, execute_research, write_output_bytes
from ._generated.models import (
    AutoReloadRequest,
    CheckoutRequest,
    LaunchRequest,
    MintApiKeyRequest,
    Money,
    RunApprovalDecisionRequest,
    RunLabelsRequest,
    SetSpendAlertsRequest,
    SubscriptionChangeRequest,
    UpdateApiKeyRequest,
    UploadCreateRequest,
)
from ._validation import safe_validation_detail

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

    from .aclient import AsyncSynthClient

__all__ = ["configure", "execute_resource"]

M = TypeVar("M", bound=BaseModel)

_RESOURCE_GROUPS = {
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
_IDENTIFIER_ACTIONS = frozenset(
    {"get", "finalize", "update", "rotate", "revoke", "checkout-status", "usage-export-status"}
)


def configure(
    parsers: Mapping[str, argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    """Register resource CLI subparsers over public SDK groups."""
    for group, actions in _RESOURCE_GROUPS.items():
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
            if action in {"prepare", "mint", "update"}:
                child.add_argument("--body", required=True, metavar="PATH")
            if action in {"checkout", "change-plan", "set-auto-reload", "set-spend-alerts"}:
                child.add_argument("--body", metavar="PATH")
            if action == "checkout":
                child.add_argument("--plan", dest="plan_id")
                child.add_argument("--pack", dest="pack_id")
                child.add_argument("--amount", help="custom credit amount in USD (e.g. 25.00)")
            if action == "change-plan":
                child.add_argument("--plan", dest="plan_id")
            if action == "set-auto-reload":
                mode = child.add_mutually_exclusive_group()
                mode.add_argument(
                    "--enable", dest="reload_enabled", action="store_true", default=None
                )
                mode.add_argument("--disable", dest="reload_enabled", action="store_false")
                child.add_argument("--threshold", help="trigger balance in USD (e.g. 10.00)")
                child.add_argument("--amount", help="reload amount in USD (e.g. 25.00)")
                child.add_argument("--disclosure-version")
            if action == "set-spend-alerts":
                child.add_argument("--monthly-amount", help="monthly spend alert threshold in USD")
                child.add_argument(
                    "--clear", action="store_true", help="disable monthly spend alert"
                )
            if action in _IDENTIFIER_ACTIONS and (
                group != "organization" or action == "usage-export-status"
            ):
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
    configure_research(parsers, common)


def _read_body_bytes(path: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"--body unreadable: {path}: {type(error).__name__}") from error


async def _parse_body(model: type[M], path: str) -> M:
    raw = await asyncio.to_thread(_read_body_bytes, path)
    try:
        return model.model_validate_json(raw)
    except ValidationError as error:
        raise ValueError(
            f"--body invalid {model.__name__}: {safe_validation_detail(error)}"
        ) from None


async def _execute_inspect(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
    match args.view:
        case "program":
            return await client.swarms.program(args.run_id)
        case "configuration":
            return await client.swarms.configuration(args.run_id)
        case "sandboxes":
            return await client.swarms.sandboxes(args.run_id)
        case "wait-update":
            if args.timeout_seconds <= 0:
                raise ValueError("--timeout-seconds must be positive")
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


async def _execute_keys(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel | None:
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
            await client.account.api_keys.revoke(args.identifier)
            return None
        case "mint":
            key = await _parse_body(MintApiKeyRequest, args.body)
            return await client.account.api_keys.mint(
                purpose=key.purpose,
                name=key.name,
                scopes=key.scopes,
                expires_at=key.expires_at,
                request_id=key.request_id,
                idempotency_key=args.idempotency_key,
            )
        case "update":
            grants = await _parse_body(UpdateApiKeyRequest, args.body)
            return await client.account.api_keys.update_grants(
                args.identifier,
                scopes=grants.scopes,
                expected_version=grants.expected_version,
                name=grants.name,
            )
        case _:
            raise ValueError("unknown keys action")


async def _execute_organization(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel:
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


async def _execute_billing(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel | None:
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
            return await client.billing.cancel_subscription(idempotency_key=args.idempotency_key)
        case "resume-subscription":
            return await client.billing.resume_subscription(idempotency_key=args.idempotency_key)
        case "preview-plan":
            return await client.billing.preview_subscription_change(args.plan_id)
        case "payment-method":
            return await client.billing.payment_method_session(idempotency_key=args.idempotency_key)
        case "auto-reload":
            return await client.billing.auto_reload()
        case "spend-alerts":
            return await client.billing.spend_alerts()
        case "checkout":
            if args.body is not None:
                if args.plan_id is not None or args.pack_id is not None or args.amount is not None:
                    raise ValueError("pass either --body or --plan/--pack/--amount, not both")
                checkout = await _parse_body(CheckoutRequest, args.body)
                return await client.billing.checkout(
                    plan_id=checkout.plan_id,
                    pack_id=checkout.pack_id,
                    credit=checkout.credit,
                    idempotency_key=args.idempotency_key,
                )
            chosen = sum(x is not None for x in (args.plan_id, args.pack_id, args.amount))
            if chosen != 1:
                raise ValueError(
                    "checkout requires --body PATH or exactly one of --plan, --pack, or --amount"
                )
            credit = Money(amount=args.amount, currency="USD") if args.amount is not None else None
            return await client.billing.checkout(
                plan_id=args.plan_id,
                pack_id=args.pack_id,
                credit=credit,
                idempotency_key=args.idempotency_key,
            )
        case "change-plan":
            if args.body is not None:
                if args.plan_id is not None:
                    raise ValueError("pass either --body or --plan, not both")
                plan = await _parse_body(SubscriptionChangeRequest, args.body)
                plan_id = str(plan.plan_id)
            elif args.plan_id is not None:
                plan_id = str(args.plan_id)
            else:
                raise ValueError("change-plan requires --plan PLAN_ID or --body PATH")
            return await client.billing.change_subscription_plan(
                plan_id, idempotency_key=args.idempotency_key
            )
        case "set-auto-reload":
            if args.body is not None:
                reload = await _parse_body(AutoReloadRequest, args.body)
                return await client.billing.set_auto_reload(
                    enabled=reload.enabled,
                    threshold=reload.threshold,
                    amount=reload.amount,
                    disclosure_version=reload.disclosure_version,
                    idempotency_key=args.idempotency_key,
                )
            if args.reload_enabled is None:
                raise ValueError("set-auto-reload requires --enable, --disable, or --body PATH")
            threshold = (
                Money(amount=args.threshold, currency="USD") if args.threshold is not None else None
            )
            amount = Money(amount=args.amount, currency="USD") if args.amount is not None else None
            return await client.billing.set_auto_reload(
                enabled=args.reload_enabled,
                threshold=threshold,
                amount=amount,
                disclosure_version=args.disclosure_version,
                idempotency_key=args.idempotency_key,
            )
        case "set-spend-alerts":
            if args.body is not None:
                alerts = await _parse_body(SetSpendAlertsRequest, args.body)
                return await client.billing.set_spend_alerts_request(
                    alerts, idempotency_key=args.idempotency_key
                )
            if args.clear and args.monthly_amount is not None:
                raise ValueError("pass either --monthly-amount or --clear, not both")
            if not args.clear and args.monthly_amount is None:
                raise ValueError(
                    "set-spend-alerts requires --monthly-amount, --clear, or --body PATH"
                )
            monthly = None if args.clear else Money(amount=args.monthly_amount, currency="USD")
            return await client.billing.set_spend_alerts(
                monthly_amount=monthly, idempotency_key=args.idempotency_key
            )
        case _:
            raise ValueError("unknown billing action")


async def execute_resource(client: AsyncSynthClient, args: argparse.Namespace) -> BaseModel | None:
    """Execute one resource, inspect, labels, content, or research CLI subcommand."""
    match args.command:
        case "labels":
            labels = await _parse_body(RunLabelsRequest, args.body)
            return await client.swarms.update_labels(
                args.run_id, labels.labels, expected_revision=labels.expected_revision
            )
        case "content":
            content = await client.swarms.output_content(
                args.run_id, args.output_id, revision_id=args.revision, digest_sha256=args.sha256
            )
            written = await asyncio.to_thread(write_output_bytes, args.out, content)
            return RootModel[str](written)
        case "families":
            return await client.swarms.families()
        case "inspect":
            return await _execute_inspect(client, args)
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
                    body = await _parse_body(UploadCreateRequest, args.body)
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
            return await _execute_keys(client, args)
        case "organization":
            return await _execute_organization(client, args)
        case "billing":
            return await _execute_billing(client, args)
        case "runs":
            match args.runs_action:
                case "preflight":
                    launch = await _parse_body(LaunchRequest, args.body)
                    return await client.swarms.preflight_request(launch)
                case "approvals":
                    return await client.swarms.approvals(
                        args.run_id, status=args.status, cursor=args.cursor, limit=args.limit
                    )
                case "decide-approval":
                    decision = await _parse_body(RunApprovalDecisionRequest, args.body)
                    return await client.swarms.decide_approval(
                        args.run_id,
                        args.approval_id,
                        decision,
                        idempotency_key=args.idempotency_key,
                    )
                case _:
                    raise ValueError("unknown runs action")
        case "artifacts" | "experiments" | "research-log":
            return await execute_research(client, args)
        case _:
            raise ValueError("unknown resource command")
