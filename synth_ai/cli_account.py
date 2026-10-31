"""Basic cloud account companion, without workload or purchasing commands."""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

from synth_ai.auth.oauth import AuthRefusalError
from synth_ai.client import SynthClient, resolve_url
from synth_ai.core.http.request import validate_base_url
from synth_ai.options import ReadOptions
from synth_ai.types.account_cloud import (
    AccountBalance,
    AccountContext,
    AccountLimits,
    AccountUsageEventPage,
    AccountUsageSummary,
    CreditAlertState,
    CurrentPlan,
)
from synth_ai.types.money import Money


def _text(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def _money(value: Money | None) -> str:
    if value is None:
        return "Unknown"
    with localcontext() as context:
        context.prec = 64
        exact = Decimal(value.coefficient).scaleb(-value.scale)
        return "$" + format(exact, "f")


def _window(args):
    if bool(args.period_start) != bool(args.period_end):
        raise ValueError("Both --from and --to are required together")
    if args.period_start:
        return datetime.fromisoformat(args.period_start), datetime.fromisoformat(args.period_end)
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    return end - timedelta(days=args.days), end


def _dashboard_path(path: str) -> str:
    origin = os.environ.get("SYNTH_DASHBOARD_URL")
    if origin is None:
        if resolve_url(None) != "https://api.usesynth.ai":
            raise AuthRefusalError(
                "dashboard_origin_required",
                400,
                service="synth-ai",
                route="account",
                cause="Set SYNTH_DASHBOARD_URL for the selected API origin",
            )
        origin = "https://www.usesynth.ai"
    return validate_base_url(origin) + path


def run_account(args: argparse.Namespace) -> int:
    command = args.account_command
    if command == "limits" and args.candidate_manifest_sha256 is not None:
        raise AuthRefusalError("account_candidate_manifest_unsupported", 400, service="synth-ai")
    if command in {"billing", "keys", "apps"}:
        url = _dashboard_path(
            {"billing": "/billing", "keys": "/keys", "apps": "/settings/connected-apps"}[command]
        )
        if not args.no_browser:
            # The public URL is still returned, including on headless hosts.
            with suppress(webbrowser.Error, OSError):
                webbrowser.open(url)
        origin = resolve_url(None)
        print(
            json.dumps({"url": url, "api_origin": origin})
            if args.json
            else f"{url}\nAPI: {_text(origin)}"
        )
        return 0
    value: (
        CreditAlertState
        | AccountLimits
        | AccountContext
        | CurrentPlan
        | AccountBalance
        | AccountUsageSummary
        | AccountUsageEventPage
    )
    read = ReadOptions(read_retry_count=0)
    with SynthClient() as client:
        if command == "context":
            value = client.account.context(request=read)
        elif command == "limits":
            value = client.account.limits(request=read)
        elif command == "plan":
            value = client.account.plan(request=read)
        elif command == "alerts":
            value = client.account.alerts(request=read)
        elif command == "balance":
            value = client.account.balance(request=read)
        elif command == "usage":
            value = client.account.usage(*_window(args), product=args.product, request=read)
        elif command == "logs":
            value = client.account.events(
                *_window(args),
                product=args.product,
                cursor=args.cursor,
                limit=args.limit,
                request=read,
            )
        else:
            raise ValueError("Unsupported account command")
    if args.json:
        print(value.model_dump_json())
        return 0
    if isinstance(value, AccountContext):
        print(
            f"Organization\t{_text(value.org_id)}\nUser\t{_text(value.user_id)}\nAPI\t{_text(resolve_url(None))}\nCredential\t{value.credential_kind}\nCredential ID\t{_text(value.credential_id or 'Unknown')}\nPermissions\t{', '.join(value.scopes)}"
        )
    elif isinstance(value, CurrentPlan):
        print(f"Plan\t{_text(value.name or value.state)}")
        if value.state == "ready":
            print(
                f"Monthly fee\t{_money(value.monthly_price)}\nIncluded / 7 days\t{_money(value.weekly_included)}\nRemaining included\t{_money(value.remaining_this_week)}\nResets (UTC)\t{value.resets_at.isoformat() if value.resets_at else 'Unknown'}"
            )
        if value.provider_observed_at:
            print(f"Plan checked (UTC)\t{value.provider_observed_at.isoformat()}")
        if value.wallet_observed_at:
            print(f"Balance updated (UTC)\t{value.wallet_observed_at.isoformat()}")
        if value.renews_at:
            print(f"Renews (UTC)\t{value.renews_at.isoformat()}")
        if value.cancels_at:
            print(f"Ends (UTC)\t{value.cancels_at.isoformat()}")
    elif isinstance(value, AccountLimits):
        print("Product\tLimit\tCapacity\tUsed\tRemaining\tReset")
        for limit in value.limits:
            capacity = (
                "Unavailable"
                if limit.state == "unavailable"
                else "Unlimited"
                if limit.value is None
                else str(limit.value)
            )
            print(
                "\t".join(
                    (
                        limit.product,
                        limit.limit_id,
                        capacity,
                        str(limit.used) if limit.used is not None else "Unknown",
                        str(limit.remaining) if limit.remaining is not None else "Unknown",
                        limit.resets_at.isoformat() if limit.resets_at else "Unknown",
                    )
                )
            )
        for name in (
            "max_results",
            "max_excerpts_per_result",
            "query_max_bytes",
            "contents_max_bytes",
        ):
            print(f"Request {name}\t{getattr(value.request_limits, name)}")
    elif isinstance(value, CreditAlertState):
        print(
            f"Low-credit alerts\t{value.state}\nThreshold\t${value.threshold_dollars}\nRecipient\t{_text(value.recipient or 'None')}"
        )
        if value.last_sent_at:
            print(f"Last sent (UTC)\t{value.last_sent_at.isoformat()}")
    elif isinstance(value, AccountBalance):
        print(
            f"Available\t{_money(value.total.available)}\nReserved\t{_money(value.total.reserved)}"
        )
        for part in value.classes:
            print(f"{part.funding_class.title()}\t{_money(part.available)}")
    elif isinstance(value, AccountUsageSummary):
        print("Product\tOperations\tMeasured charge\tEstimated charge\tPending\tUnknown")
        by_product = {item.product: item.totals for item in value.products}
        for source in value.sources:
            total = by_product.get(source.product)
            if source.state == "unavailable" or total is None:
                print(f"{source.product}\tUnavailable\tUnknown\tUnknown\tUnknown\tUnknown")
                continue
            charge = _money(total.measured_charge)
            estimate = _money(total.estimated_charge) if total.estimated_charge_item_count else "—"
            if total.item_count and total.included_item_count == total.item_count:
                charge = "Included"
            elif total.measured_charge_item_count == 0 and total.estimated_charge_item_count:
                charge = "—"
            if total.measured_charge_item_count == 0 and (
                total.pending_charge_item_count or total.unknown_charge_item_count
            ):
                charge = "Pending" if total.pending_charge_item_count else "Unknown"
            product_label = source.product + (" (partial)" if source.state == "partial" else "")
            count = str(total.item_count)
            if source.state == "partial" and total.item_count == 0:
                count, charge = "Pending", "Updating"
            print(
                f"{product_label}\t{count}\t{charge}\t{estimate}\t{total.pending_charge_item_count}\t{total.unknown_charge_item_count}"
            )
        print(f"Period\t{value.period_start.isoformat()} to {value.period_end.isoformat()}")
    else:
        print("Recorded (UTC)\tProduct\tEvent\tReceipt\tState")
        for item in value.items:
            state = (
                item.facts.get("reason_code")
                or item.facts.get("cost_state")
                or item.facts.get("quality")
                or item.facts.get("settlement_state")
                or "Recorded"
            )
            print(
                f"{item.recorded_at.isoformat()}\t{item.product}\t{item.source}\t{_text(item.source_id)}\t{_text(state)}"
            )
        for source in value.sources:
            print(
                f"Coverage\t{source.product}\t{source.state}"
                + (f"\t{_text(source.code)}" if source.code else "")
            )
        if value.next_cursor:
            print(f"Next cursor\t{_text(value.next_cursor)}")
    if isinstance(value, AccountUsageSummary):
        for source in value.sources:
            print(
                f"{source.product} coverage\t{source.state}\t{source.projected_at.isoformat() if source.projected_at else 'Unknown'}\t{source.watermark or 'Unknown'}"
            )
            total = next(
                (part.totals for part in value.products if part.product == source.product), None
            )
            if total is not None:
                for metric in total.metrics:
                    print(
                        f"{source.product} {_text(metric.name)}\t{metric.quantity if metric.quantity is not None else metric.state} {_text(metric.unit)}"
                    )
    print(f"As of (UTC)\t{value.observed_at.isoformat()}")
    return 0


def add_account_commands(commands) -> None:
    account = commands.add_parser("account").add_subparsers(dest="account_command", required=True)
    for name in (
        "limits",
        "context",
        "plan",
        "balance",
        "alerts",
        "usage",
        "logs",
        "billing",
        "keys",
        "apps",
    ):
        command = account.add_parser(name)
        command.add_argument("--json", action="store_true")
        if name == "limits":
            command.add_argument("--candidate-manifest-sha256")
        if name in {"billing", "keys", "apps"}:
            command.add_argument("--no-browser", action="store_true")
        if name in {"usage", "logs"}:
            command.add_argument(
                "--days", type=int, choices=range(1, 367), default=7, metavar="1..366"
            )
            command.add_argument("--from", dest="period_start")
            command.add_argument("--to", dest="period_end")
            command.add_argument("--product", choices=("index", "swarms", "monitor", "inference"))
        if name == "logs":
            command.add_argument(
                "--limit", type=int, choices=range(1, 201), default=50, metavar="1..200"
            )
            command.add_argument("--cursor")
