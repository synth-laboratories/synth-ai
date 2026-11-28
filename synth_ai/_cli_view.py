"""Human-readable CLI rendering and the per-run event checkpoint store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ._cli_checkpoint import checkpoint_path as checkpoint_path
from ._cli_checkpoint import load_checkpoint as load_checkpoint
from ._cli_checkpoint import save_checkpoint as save_checkpoint
from ._cli_checkpoint import state_dir as state_dir
from ._cli_table import facts as facts
from ._cli_table import money as money
from ._cli_table import table as table
from ._cli_table import with_next_cursor as _with_next_cursor
from ._cli_watch import milestones_from_snapshot as milestones_from_snapshot
from ._cli_watch import watch_item as watch_item
from ._generated.billing_plans import PAYG_DAILY_SPEND_USD, PAYG_WEEKLY_SPEND_USD
from ._generated.models import (
    Account,
    ApiVersionInfo,
    AutoReload,
    Balance,
    BillingHistoryPage,
    BillingStatementPage,
    EventPage,
    FamilyList,
    File,
    LaunchChoices,
    LimitList,
    LimitMeter,
    OAuthGrantRevoked,
    OutputPage,
    PlanCatalog,
    PlanRequest,
    PlanRequestList,
    PromoRedemption,
    RunAllocation,
    RunAllocationChoice,
    RunCharge,
    RunControlReceipt,
    RunEvent,
    RunLimitsProgress,
    RunMessage,
    RunMessagePage,
    RunOutput,
    RunPage,
    RunProfileCatalog,
    RunReport,
    RunResult,
    RunSnapshot,
    RunTaskPage,
    SpendAlerts,
    Subscription,
    UsageRecordPage,
    UsageSummary,
)
from ._version import __version__
from .topologies import topology_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._auth_models import AuthStatus


class SavedOutput(BaseModel):
    """One downloaded run output and where it was written."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", hide_input_in_errors=True)

    output: RunOutput
    path: str


def snapshot(value: RunSnapshot) -> str:
    rows: list[tuple[str, object]] = [
        ("run", value.run_id),
        ("state", value.state),
        ("profile", value.profile.id if value.profile is not None else None),
        ("model", value.model),
        ("result", "available" if value.result_available else "pending"),
        ("created", value.created_at),
        ("closed", value.closed_at),
    ]
    # Contract 1.32.0: only a run waiting for capacity carries its queue standing.
    if value.queue is not None:
        rows += [
            ("queue", f"position {value.queue.position} ({value.queue.ahead} ahead)"),
            ("starts", value.queue.estimated_start_at),
            ("expires", value.queue.expires_at),
        ]
    return facts(rows)


def receipt(value: RunControlReceipt) -> str:
    return facts([("run", value.resource_id), ("command", value.command), ("state", value.state)])


def event(value: RunEvent) -> str:
    return "  ".join([str(value.sequence).rjust(6), value.observed_at or "-", value.kind])


def message(value: RunMessage) -> str:
    """Keep retention unavailability distinct from a blank message."""
    text = (
        value.body.text
        if value.body.kind == "available"
        else "message no longer available (retention)"
    )
    return "  ".join([value.message_id, value.author, value.state.kind, text])


def messages(value: RunMessagePage) -> str:
    if value.availability.kind == "unavailable":
        return value.availability.reason
    return _with_next_cursor(
        "\n".join(message(item) for item in value.items),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def events(value: EventPage) -> str:
    return _with_next_cursor(
        "\n".join(event(item) for item in value.items),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def runs(value: RunPage) -> str:
    return _with_next_cursor(
        table(
            ["RUN", "STATE", "PROFILE", "CREATED"],
            [
                [r.run_id, r.state, r.profile.id if r.profile is not None else None, r.created_at]
                for r in value.items
            ],
        ),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def tasks(value: RunTaskPage) -> str:
    return _with_next_cursor(
        table(["TASK", "STATE"], [[t.task_id, t.state] for t in value.items]),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def outputs(value: OutputPage) -> str:
    return _with_next_cursor(
        table(
            ["OUTPUT", "STATE", "FILENAME"],
            [[o.output_id, o.state, o.filename] for o in value.items],
        ),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def profiles(value: RunProfileCatalog) -> str:
    return table(
        ["PROFILE", "NAME", "AVAILABLE", "RESEARCH TASK", "DEFAULT"],
        [
            [
                p.id,
                p.name,
                "yes" if p.available else "no",
                "required" if p.requires_research_task else "-",
                "yes" if p.id == value.default_profile else "",
            ]
            for p in value.profiles
        ],
    )


def families(value: FamilyList) -> str:
    parts: list[str] = []
    if value.topologies is not None and len(value.topologies) > 0:
        parts.append(
            table(
                ["TOPOLOGY", "ID", "AVAILABILITY"],
                [
                    [
                        topology_name(entry),
                        f"{entry.topology}@{entry.version}",
                        entry.availability.value,
                    ]
                    for entry in value.topologies
                ],
            )
        )
    if value.families:
        parts.append(
            table(
                ["FAMILY", "VERSION", "BINDING", "ACTORS", "SUMMARY"],
                [
                    [
                        family.id,
                        family.version,
                        family.research_binding,
                        ", ".join(family.required_actor_types) or "-",
                        family.summary,
                    ]
                    for family in value.families
                ],
            )
        )
    return "\n\n".join(parts) if parts else facts([("families", "none")])


def promo_redemption(value: PromoRedemption) -> str:
    rows: list[tuple[str, object]] = [
        ("redemption", value.redemption_id),
        ("campaign", value.campaign_id),
        ("redeemed", value.redeemed_at),
        ("replayed", "yes" if value.replayed else "no"),
        ("grants", ", ".join(value.grant_ids) or "-"),
    ]
    if value.override_ids:
        rows.append(("overrides", ", ".join(value.override_ids)))
    if value.trial is not None:
        rows.append(
            (
                "trial",
                f"{value.trial.plan_id!s} ({value.trial.trial_id}, ends {value.trial.ends_at})",
            )
        )
    return facts(rows)


def files(value: Sequence[File]) -> str:
    return table(["FILE", "NAME", "BYTES"], [[f.file_id, f.name, f.size_bytes] for f in value])


def saved(value: Sequence[SavedOutput]) -> str:
    return table(["OUTPUT", "PATH"], [[s.output.output_id, s.path] for s in value])


def launch_choices(value: LaunchChoices) -> str:
    text = facts(
        [
            ("menu revision", value.menu_revision),
            ("run profile", value.run_profile),
            ("domicile", value.domicile_requirement.value),
            ("affordability", value.affordability.status.value),
        ]
    )
    text += "\n\n" + table(
        ["ACTOR", "ROLE", "DEFAULT INFERENCE", "DEFAULT SANDBOX", "PLANNER MAY DECIDE"],
        [
            [
                r.actor_id,
                r.role.value,
                r.default_inference_profile,
                r.default_sandbox_profile or "planner decides",
                "yes" if r.planner_decides_allowed else "no",
            ]
            for r in value.roles
        ],
    )
    text += "\n\n" + table(
        ["INFERENCE", "MODEL", "EFFORT", "INPUT /1M", "OUTPUT /1M", "PER CALL CAP"],
        [
            [
                p.profile_id,
                p.model,
                p.effort,
                money(p.input_price_per_million_tokens),
                money(p.output_price_per_million_tokens),
                money(p.per_call_spend_cap),
            ]
            for p in value.inference_profiles
        ],
    )
    has_gpu = any(s.gpu_kind is not None or s.gpu_count is not None for s in value.sandbox_profiles)
    sandbox_header = [
        "SANDBOX",
        "SIZE",
        "DOMICILE",
        *(["GPU"] if has_gpu else []),
        "HOURLY",
        "OPTIONS",
    ]
    sandbox_rows: list[list[object]] = []
    for s in value.sandbox_profiles:
        row: list[object] = [s.profile_id, s.size_id, s.domicile.value]
        if has_gpu:
            if s.gpu_kind is not None and s.gpu_count is not None:
                gpu_spec: str | None = f"{s.gpu_count}x {s.gpu_kind}"
            elif s.gpu_kind is not None:
                gpu_spec = s.gpu_kind
            elif s.gpu_count is not None:
                gpu_spec = f"{s.gpu_count}x"
            else:
                gpu_spec = None
            row.append(gpu_spec)
        row.extend(
            [
                money(s.hourly_price),
                ("choose one of " if s.option_required else "") + str(len(s.options)),
            ]
        )
        sandbox_rows.append(row)
    text += "\n\n" + table(sandbox_header, sandbox_rows)
    option_rows = [
        [
            s.profile_id,
            opt.option_id,
            money(opt.hourly_price),
            "yes" if s.option_required else "no",
        ]
        for s in value.sandbox_profiles
        for opt in s.options
    ]
    if option_rows:
        text += "\n\n" + table(
            ["SANDBOX", "OPTION", "HOURLY", "REQUIRED"],
            option_rows,
        )
    if value.exclusions:
        text += "\n\n" + table(
            ["EXCLUDED", "KIND", "OPTION", "CAUSE"],
            [[e.profile_id, e.kind.value, e.option_id, e.cause.value] for e in value.exclusions],
        )
    return text


def _choice(value: RunAllocationChoice) -> str:
    chosen = "planner decides" if value.profile_id is None else value.profile_id
    return chosen + " (" + value.source.value + ")"


def allocation(value: RunAllocation) -> str:
    text = facts(
        [
            ("run", value.run_id),
            ("menu revision", value.menu_revision),
            ("final", "yes" if value.final else "no"),
        ]
    )
    if value.actors:
        text += "\n\n" + table(
            ["ACTOR", "INFERENCE", "SANDBOX", "PLACED", "REPLACED", "CALLS", "COMPUTE"],
            [
                [
                    a.actor_id,
                    _choice(a.launch.inference),
                    _choice(a.launch.sandbox),
                    a.placements[-1].profile_id if a.placements else None,
                    a.replacements,
                    a.usage.calls,
                    money(a.compute_charge)
                    if a.compute_charge is not None
                    else a.compute_metering.value,
                ]
                for a in value.actors
            ],
        )
    return text


def report(value: RunReport) -> str:
    rows: list[tuple[str, object]] = [
        ("run", value.run_id),
        ("settlement", value.settlement.status.value),
        ("charge", money(value.settlement.total)),
        ("reward", value.reward.status.value),
        ("teardown", value.teardown.value),
    ]
    text = facts(rows)
    if value.by_model:
        text += "\n\n" + table(
            ["PROVIDER", "MODEL", "CALLS", "CHARGE"],
            [[m.provider, m.model, m.calls, money(m.charge)] for m in value.by_model],
        )
    if value.sandboxes:
        text += "\n\n" + table(
            ["ACTOR", "SIZE", "DOMICILE", "READY MS", "COMPUTE"],
            [
                [
                    item.profile.actor_id if item.profile is not None else item.role.value,
                    item.profile.size_id
                    if item.profile is not None
                    else f"{item.cpu_millis / 1000:g} CPU / {item.memory_bytes / (1 << 30):g} GiB",
                    item.profile.domicile.value if item.profile is not None else "unknown",
                    item.time_to_ready_ms if item.time_to_ready_ms is not None else "unknown",
                    money(item.compute_charge)
                    if item.compute_charge is not None
                    else item.compute_metering.value,
                ]
                for item in value.sandboxes
            ],
        )
    return text


def charge_line(value: RunCharge | None) -> str | None:
    """One closing line: the run's charge and whether it is final."""
    if value is None:
        return None
    return "charge  " + money(value.amount) + (" (final)" if value.final else " (not final)")


def result(value: RunResult) -> str:
    """The answer text alone, so stdout can be captured; the CLI prints the charge on stderr."""
    if value.answer is not None:
        return value.answer
    summary = facts(
        [
            ("run", value.run.run_id),
            ("state", value.run.state),
            ("answer", "none (no available text output or objective answer)"),
        ]
    )
    return summary


def status_line(value: RunResult) -> str:
    """The run's authoritative terminal outcome (completed, partial, failed or cancelled)."""
    return "status  " + value.status


def accounting_line(value: RunResult) -> str:
    """The final ledger-backed charge, or that Core has no accounting reader for it."""
    if value.accounting is None:
        return "accounting  unavailable"
    return "accounting  " + money(value.accounting.amount) + " (final)"


def result_file_line(value: RunResult) -> str | None:
    """The run's result.md output, read with `download RUN OUTPUT`; None when unavailable."""
    if value.result_file is None:
        return None
    return "result.md  " + value.result_file.output_id


def result_json(value: RunResult) -> str:
    """`RunResult` as Core returned it: run_id, status, accounting, the answer and the charge."""
    return value.model_dump_json(exclude_unset=True)


def usage_summary(value: UsageSummary) -> str:
    rows: list[tuple[str, object]] = [
        ("since", value.since),
        ("until", value.until),
    ]
    if value.data_through is not None:
        rows.append(("data through", value.data_through))
    rows.extend(
        [
            ("charge", money(value.total.charge)),
            ("pending", f"{money(value.pending.amount)} ({value.pending.count})"),
        ]
    )
    settling = getattr(value, "settling", None)
    if settling is not None:
        rows.append(("settling", f"{money(settling.amount)} ({settling.count})"))
    rows.extend(
        [
            ("input tokens", value.total.input_tokens),
            ("output tokens", value.total.output_tokens),
            ("records", value.total.records),
        ]
    )
    funding = getattr(value, "funding", None)
    if funding is not None:
        rows.extend(
            [
                ("plan", funding.plan),
                ("available", money(funding.available)),
                ("purchased credits", money(funding.purchased_credits)),
                ("held", money(funding.held)),
            ]
        )
        if funding.included_allowance is not None:
            ia = funding.included_allowance
            rows.append(
                (
                    "weekly allowance",
                    f"{money(ia.remaining)} / {money(ia.limit)} remaining (resets {ia.reset_at})",
                )
            )
    return facts(rows)


def usage_records(value: UsageRecordPage) -> str:
    return _with_next_cursor(
        table(
            ["RECORD", "KIND", "MODEL", "INPUT", "OUTPUT", "CHARGE"],
            [
                [
                    r.record_id,
                    r.kind.value,
                    r.preset,
                    r.input_tokens,
                    r.output_tokens,
                    money(r.charge),
                ]
                for r in value.items
            ],
        ),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def balance(value: Balance) -> str:
    rows: list[tuple[str, object]] = [
        ("plan", value.plan),
        ("available", money(value.available)),
        ("purchased credits", money(value.purchased_credits)),
        ("promotional credits", money(value.promotional_credits)),
        ("operator credits", money(value.operator_credits)),
        ("debt", money(value.debt)),
        ("held", money(value.held)),
        ("headroom", money(value.headroom)),
        ("ledger", money(value.ledger)),
    ]
    if value.payg_credits_active:
        rows.insert(
            1,
            (
                "spend tier",
                f"payg_credits (${PAYG_DAILY_SPEND_USD}/day, ${PAYG_WEEKLY_SPEND_USD}/week)",
            ),
        )
    if value.included_allowance is not None:
        ia = value.included_allowance
        rows.append(
            (
                "weekly allowance",
                f"{money(ia.remaining)} / {money(ia.limit)} remaining (resets {ia.reset_at})",
            )
        )
    return facts(rows)


def plans(value: PlanCatalog) -> str:
    plan_table = table(
        ["PLAN", "NAME", "KIND", "PRICE", "WEEKLY INCLUDED", "CHECKOUT"],
        [
            [
                p.plan_id,
                p.name,
                p.kind.value,
                money(p.price),
                money(p.weekly_included),
                "ready" if p.checkout_available else p.price_state.value,
            ]
            for p in value.plans
        ],
    )
    pack_rows = [
        [pack.pack_id, pack.name, money(pack.price), "ready" if pack.checkout_available else "off"]
        for pack in value.credit_packs
    ]
    parts = [plan_table]
    if pack_rows:
        parts.append(table(["PACK", "NAME", "PRICE", "CHECKOUT"], pack_rows))
    parts.append(
        facts(
            [
                ("custom credit min", money(value.custom_credit.min)),
                ("custom credit max", money(value.custom_credit.max)),
                ("switchable plans", ", ".join(value.switchable_plans) or "-"),
            ]
        )
    )
    return "\n\n".join(parts)


def billing_history(value: BillingHistoryPage) -> str:
    return _with_next_cursor(
        table(
            ["ENTRY", "CAUSE", "FUNDING", "DELTA", "HELD DELTA", "CREATED"],
            [
                [
                    e.entry_id,
                    e.cause.value if hasattr(e.cause, "value") else e.cause,
                    e.funding_class or "-",
                    money(e.delta),
                    money(e.held_delta),
                    e.created_at,
                ]
                for e in value.items
            ],
        ),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def billing_statements(value: BillingStatementPage) -> str:
    return _with_next_cursor(
        table(
            ["MONTH", "OPENING", "PURCHASES", "ALLOWANCE", "CHARGES", "CLOSING"],
            [
                [
                    s.month,
                    money(s.opening_ledger),
                    money(s.purchases),
                    money(s.allowance),
                    money(s.charges),
                    money(s.closing_ledger),
                ]
                for s in value.items
            ],
        ),
        has_more=value.has_more,
        next_cursor=value.next_cursor,
    )


def subscription(value: Subscription | None) -> str:
    if value is None:
        return facts([("subscription", "none")])
    return facts(
        [
            ("plan", value.plan_id),
            ("status", value.status.value if hasattr(value.status, "value") else value.status),
            ("period end", value.current_period_end),
            ("cancel at period end", value.cancel_at_period_end),
            ("cancel at", value.cancel_at),
        ]
    )


def auto_reload(value: AutoReload) -> str:
    return facts(
        [
            ("enabled", value.enabled),
            ("threshold", money(value.threshold)),
            ("amount", money(value.amount)),
        ]
    )


def spend_alerts(value: SpendAlerts) -> str:
    return facts(
        [
            ("monthly amount", money(value.monthly_amount)),
            ("last notified", value.last_notified),
        ]
    )


def limits(value: LimitList) -> str:
    header: list[tuple[str, object]] = [("plan", value.plan)]
    if value.payg_credits_active:
        header.append(("spend tier", "payg_credits"))
    return facts(header) + (
        "\n\n"
        + table(
            ["RESOURCE", "METRIC", "WINDOW", "CURRENT", "CAP", "RESET AT"],
            [[i.resource, i.metric, i.window, i.current, i.cap, i.reset_at] for i in value.items],
        )
        if value.items
        else ""
    )


def plan_request(value: PlanRequest) -> str:
    return facts(
        [
            ("request", value.request_id),
            ("state", value.state),
            ("created", value.created_at),
            ("decided", value.decided_at),
            ("reason", value.decision_reason),
        ]
    )


def oauth_grant_revoked(value: OAuthGrantRevoked) -> str:
    return facts([("grant", value.grant_id), ("status", value.status), ("outcome", value.outcome)])


def plan_requests(value: PlanRequestList) -> str:
    return table(
        ["REQUEST", "STATE", "CREATED", "DECIDED", "REASON"],
        [
            [i.request_id, i.state, i.created_at, i.decided_at, i.decision_reason]
            for i in value.items
        ],
    )


def account(value: Account) -> str:
    rows: list[tuple[str, object]] = [
        ("org", value.org.name),
        ("org id", value.org.id),
        ("plan", value.org.plan),
        ("access tier", value.org.access_tier),
        ("user", value.principal.user_id),
        ("credential", value.principal.credential_kind.value),
    ]
    if value.billing_attention is not None:
        rows.append(("billing attention", value.billing_attention.kind))
    if value.promotional_grants is not None and len(value.promotional_grants) > 0:
        rows.append(("promotional grants", len(value.promotional_grants)))
    return facts(rows)


def render_billing(action: str, value: BaseModel | None) -> str | None:
    """Format a billing resource response for human output, or return None to fall back to JSON."""
    if action == "balance" and isinstance(value, Balance):
        return balance(value)
    if action == "plans" and isinstance(value, PlanCatalog):
        return plans(value)
    if action == "history" and isinstance(value, BillingHistoryPage):
        return billing_history(value)
    if action == "statements" and isinstance(value, BillingStatementPage):
        return billing_statements(value)
    if action == "subscription" and (value is None or isinstance(value, Subscription)):
        return subscription(value)
    if action == "auto-reload" and isinstance(value, AutoReload):
        return auto_reload(value)
    if action == "spend-alerts" and isinstance(value, SpendAlerts):
        return spend_alerts(value)
    return None


def auth_status(value: AuthStatus) -> str:
    return facts(
        [
            ("api", value.base_url),
            ("signed in", "yes" if value.authenticated else "no"),
            ("source", value.source.value),
        ]
    )


def run_limits(value: RunLimitsProgress) -> str:
    def amount(number: int | None) -> str:
        return "unknown" if number is None else str(number)

    lines = [f"{value.run_id}  revision {value.revision}  {value.status}"]

    def rows(label: str, meters: list[LimitMeter]) -> None:
        lines.append(label)
        lines.append(
            "Limit | Cap | Used | Reserved | Unknown reserved | Unknown receipts | Remaining"
        )
        for meter in meters:
            lines.append(
                " | ".join(
                    [
                        meter.dimension,
                        amount(
                            meter.effective_cap if meter.effective_cap is not None else meter.cap
                        ),
                        amount(meter.used),
                        amount(meter.reserved),
                        amount(meter.unknown_reserved),
                        amount(meter.unknown_receipts),
                        amount(meter.remaining),
                    ]
                )
            )

    rows("Primary", value.primary)
    rows("Secondary", value.secondary)
    for actor in value.actors:
        rows(f"{actor.assignment}  {actor.state or 'unknown'}", actor.limits)
    return "\n".join(lines)


def server_version(value: ApiVersionInfo) -> str:
    return facts(
        [
            ("client", __version__),
            (
                "server",
                f"{value.server.version} ({value.server.channel}, {value.server.environment})",
            ),
            ("api", f"{value.api.version} (major {value.api.major})"),
            ("swarms api", value.api.swarms_api_version),
            (
                "client floor",
                f"min {value.client_compatibility.minimum_synth_ai_version}, "
                f"recommended {value.client_compatibility.recommended_synth_ai_version}",
            ),
            ("commit", value.build.commit),
            ("image", value.build.image_digest),
        ]
    )
