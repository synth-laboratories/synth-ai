"""In-process MockTransport server & live-slot client factory for SDK stateful PBT (SYN-6050)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx
from pydantic import JsonValue, TypeAdapter

from synth_ai import SynthClient

from .billing_ledger import BillingLedgerState, HoldRecord, make_problem
from .config import SdkPbtConfig
from .hosted_catalog import HostedOptionsEngine, cents_to_money_str, money_str_to_cents

__all__ = [
    "InProcessSdkBackend",
    "RecordedWireCall",
    "SdkPbtHarness",
    "cents_to_money_str",
    "money_str_to_cents",
]

_FIXED_TS = "2026-11-27T18:00:00Z"
_TERMINAL_RUN_STATES = frozenset({"completed", "partial", "failed", "cancelled"})
_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json_bytes(obj: JsonValue) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RecordedWireCall:
    method: str
    path: str
    status_code: int
    etag: str | None
    idempotent_replayed: bool
    raw_text: str


@dataclass
class _RunRecord:
    run_id: str
    thread_id: str
    state: str
    limits_revision: int
    maximum_charge_cents: int
    wall_time_seconds: int
    execution_closed: bool = False
    accounting_final: bool = False
    cleanup_complete: bool = False
    profile: str = "swarms-v2"
    domicile_requirement: str = "unrestricted"
    parallel_workers: int = 1
    messages: list[dict[str, JsonValue]] = field(default_factory=list)


class InProcessSdkBackend:
    """Deterministic OpenAPI-faithful mock backend invoked via httpx.MockTransport."""

    def __init__(self) -> None:
        self._seq = 0
        self._runs: dict[str, _RunRecord] = {}
        self._idempotency: dict[
            tuple[str, str, str], tuple[str, int, dict[str, str], JsonValue]
        ] = {}
        self._hosted = HostedOptionsEngine()
        self._billing = BillingLedgerState()
        self.wire_calls: list[RecordedWireCall] = []

    # Forward billing properties/helpers cleanly for SdkBillingWalletMachine
    @property
    def plan(self) -> str:
        return self._billing.plan

    @property
    def ledger_cents(self) -> int:
        return self._billing.ledger_cents

    @property
    def held_cents(self) -> int:
        return self._billing.held_cents

    @property
    def available_cents(self) -> int:
        return self._billing.available_cents

    @property
    def purchased_cents(self) -> int:
        return self._billing.purchased_cents

    @property
    def promotional_cents(self) -> int:
        return self._billing.promotional_cents

    @property
    def included_remaining_cents(self) -> int:
        return self._billing.included_remaining_cents

    @property
    def daily_spend_cents(self) -> int:
        return self._billing.daily_spend_cents

    @property
    def weekly_spend_cents(self) -> int:
        return self._billing.weekly_spend_cents

    @property
    def daily_cap_cents(self) -> int:
        return self._billing.daily_cap_cents

    @property
    def weekly_cap_cents(self) -> int:
        return self._billing.weekly_cap_cents

    @property
    def payg_credits_active(self) -> bool:
        return self._billing.payg_credits_active

    @property
    def auto_reload_enabled(self) -> bool:
        return self._billing.auto_reload_enabled

    @property
    def auto_reload_amount_cents(self) -> int:
        return self._billing.auto_reload_amount_cents

    @property
    def auto_reload_daily_count(self) -> int:
        return self._billing.auto_reload_daily_count

    @property
    def auto_reload_monthly_cents(self) -> int:
        return self._billing.auto_reload_monthly_cents

    @property
    def holds(self) -> dict[str, HoldRecord]:
        return self._billing.holds

    def deliver_stripe_webhook(
        self,
        *,
        event_id: str,
        amount_cents: int,
        payment_succeeded: bool = True,
        amount_matches_checkout: bool = True,
    ) -> tuple[int, dict[str, JsonValue], bool]:
        return self._billing.deliver_stripe_webhook(
            event_id=event_id,
            amount_cents=amount_cents,
            payment_succeeded=payment_succeeded,
            amount_matches_checkout=amount_matches_checkout,
        )

    def reserve_hold(self, amount_cents: int) -> tuple[bool, str | None, str | None]:
        return self._billing.reserve_hold(amount_cents)

    def settle_hold(self, hold_id: str, debit_cents: int) -> tuple[bool, str | None]:
        return self._billing.settle_hold(hold_id, debit_cents)

    def trigger_auto_reload_if_due(self, *, card_declined: bool = False) -> tuple[bool, str]:
        return self._billing.trigger_auto_reload_if_due(card_declined=card_declined)

    def rollover_utc_day(self) -> None:
        self._billing.rollover_utc_day()

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:022d}"

    def advance_run_state(self, run_id: str, action: str) -> None:
        rec = self._runs[run_id]
        if action == "start" and rec.state == "queued":
            rec.state = "running"
        elif action == "complete" and rec.state == "running":
            rec.state = "completed"
            rec.execution_closed = True
        elif action == "close_cancel" and rec.state == "cancelling":
            rec.state = "cancelled"
            rec.execution_closed = True
        elif action == "finalize_accounting" and rec.execution_closed:
            rec.accounting_final = True
        elif action == "complete_cleanup" and rec.execution_closed:
            rec.cleanup_complete = True

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = request.url.path
        query = dict(request.url.params)
        raw_content = request.content
        body: JsonValue = _JSON_ADAPTER.validate_json(raw_content) if raw_content else None

        if method == "GET":
            resp = self._handle_get(path, query, request.headers.get("If-None-Match"))
        else:
            resp = self._handle_mutation(
                method, path, query, body, request.headers.get("Idempotency-Key")
            )

        raw_text = resp.content.decode("utf-8") if resp.content else ""
        self.wire_calls.append(
            RecordedWireCall(
                method=method,
                path=path,
                status_code=resp.status_code,
                etag=resp.headers.get("ETag"),
                idempotent_replayed=resp.headers.get("Idempotent-Replayed", "").lower() == "true",
                raw_text=raw_text,
            )
        )
        return resp

    def _json_response(
        self,
        status: int,
        payload: JsonValue,
        *,
        extra_headers: Mapping[str, str] | None = None,
        add_etag: bool = False,
        if_none_match: str | None = None,
    ) -> httpx.Response:
        raw_bytes = _stable_json_bytes(payload)
        hdrs: dict[str, str] = {"Content-Type": "application/json", "Cache-Control": "no-store"}
        if extra_headers is not None:
            hdrs.update(extra_headers)
        if add_etag and status == 200:
            etag = f'"{_sha256_hex(raw_bytes)}"'
            hdrs["ETag"] = etag
            if if_none_match is not None and if_none_match != "":
                for cand in if_none_match.split(","):
                    cleaned = cand.strip()
                    if cleaned == "*" or cleaned.removeprefix("W/") == etag:
                        return httpx.Response(304, headers=hdrs)
        return httpx.Response(status, content=raw_bytes, headers=hdrs)

    def _handle_get(self, path: str, query: dict[str, str], inm: str | None) -> httpx.Response:
        if path == "/v1/launch-choices":
            menu = self._hosted.build_launch_menu(
                profile=query.get("profile", "swarms-v2"),
                domicile=query.get("domicile_requirement", "unrestricted"),
                parallel_workers=int(query.get("parallel_workers", "1")),
                wall_time_seconds=int(query.get("wall_time_seconds", "3600")),
                maximum_charge_cents=money_str_to_cents(query.get("maximum_charge", "5.00")),
            )
            return self._json_response(200, menu, add_etag=True, if_none_match=inm)
        if path == "/v1/billing/balance":
            return self._json_response(
                200, self._billing.balance_payload(), add_etag=True, if_none_match=inm
            )
        if path == "/v1/billing/auto-reload":
            return self._json_response(
                200, self._billing.auto_reload_payload(), add_etag=True, if_none_match=inm
            )
        if path == "/v1/billing/subscription":
            return self._json_response(
                200, self._billing.subscription_payload(), add_etag=True, if_none_match=inm
            )

        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["v1", "runs"]:
            rec = self._runs.get(parts[2])
            if rec is None:
                return self._json_response(
                    404, make_problem(404, "run_not_found", f"Run {parts[2]} not found.")
                )
            return self._json_response(
                200, self._run_snapshot_payload(rec), add_etag=True, if_none_match=inm
            )
        if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "limits":
            rec = self._runs.get(parts[2])
            if rec is None:
                return self._json_response(
                    404, make_problem(404, "run_not_found", f"Run {parts[2]} not found.")
                )
            return self._json_response(
                200, self._run_limits_payload(rec), add_etag=True, if_none_match=inm
            )
        return self._json_response(404, make_problem(404, "route_not_found", f"Unknown GET {path}"))

    def _handle_mutation(
        self,
        method: str,
        path: str,
        query: dict[str, str],
        body: JsonValue,
        idem_key: str | None,
    ) -> httpx.Response:
        body_dict = body if isinstance(body, dict) else {}
        if path == "/v1/runs/preflight":
            return self._json_response(200, self._hosted.evaluate_preflight(body_dict))
        if (
            idem_key is None
            or idem_key == ""
            or len(idem_key) > 128
            or idem_key.strip() != idem_key
        ):
            return self._json_response(
                422, make_problem(422, "mutation_identity_required", "Idempotency-Key required.")
            )
        req_hash = _sha256_hex(
            (json.dumps(sorted(query.items())) + "\x00").encode("utf-8") + _stable_json_bytes(body)
        )
        cache_key = (method, path, idem_key)
        existing = self._idempotency.get(cache_key)
        if existing is not None:
            prev_hash, prev_status, prev_headers, prev_body = existing
            if prev_hash != req_hash:
                return self._json_response(
                    409,
                    make_problem(
                        409,
                        "idempotency_key_conflict",
                        "Idempotency-Key reused with a different request payload.",
                        stage="public_idempotency",
                    ),
                )
            replay_headers = dict(prev_headers)
            replay_headers["Idempotent-Replayed"] = "true"
            return self._json_response(
                prev_status, copy.deepcopy(prev_body), extra_headers=replay_headers
            )

        resp = self._dispatch_mutation(method, path, body_dict)
        if resp.status_code < 400 or resp.status_code == 409:
            parsed: JsonValue = _JSON_ADAPTER.validate_json(resp.content) if resp.content else None
            self._idempotency[cache_key] = (
                req_hash,
                resp.status_code,
                dict(resp.headers),
                copy.deepcopy(parsed),
            )
        return resp

    def _dispatch_mutation(
        self, method: str, path: str, body: dict[str, JsonValue]
    ) -> httpx.Response:
        if method == "POST" and path == "/v1/runs":
            return self._post_create_run(body)
        if method == "POST" and path == "/v1/promotions/redeem":
            return self._json_response(*self._billing.redeem_promo(body))
        if method == "PUT" and path == "/v1/billing/auto-reload":
            return self._json_response(*self._billing.put_auto_reload(body))
        if method == "POST" and path == "/v1/billing/subscription/change":
            return self._json_response(*self._billing.change_subscription(body))

        parts = path.strip("/").split("/")
        if method == "POST" and len(parts) == 4 and parts[:2] == ["v1", "runs"]:
            run_id, action = parts[2], parts[3]
            if action in {"pause", "resume", "cancel"}:
                return self._post_run_control(run_id, action)
            if action == "limits":
                return self._post_run_limits(run_id, body)
            if action == "messages":
                return self._post_run_message(run_id, body)
        return self._json_response(
            404, make_problem(404, "route_not_found", f"Unknown {method} {path}")
        )

    def _run_snapshot_payload(self, rec: _RunRecord) -> dict[str, JsonValue]:
        return {
            "metadata": {"profile": rec.profile},
            "run_id": rec.run_id,
            "labels": {},
            "labels_revision": 1,
            "limits_revision": rec.limits_revision,
            "objectives": [],
            "guidance_coverage": [],
            "state": rec.state,
            "model": "gpt-6-luna",
            "provider": "synth-hosted",
            "result_available": rec.state == "completed",
            "execution_closed": rec.execution_closed,
            "accounting_final": rec.accounting_final,
            "cleanup_complete": rec.cleanup_complete,
            "problems": [],
            "problems_observed": False,
            "created_at": _FIXED_TS,
            "started_at": _FIXED_TS if rec.state != "queued" else None,
            "closed_at": _FIXED_TS if rec.execution_closed else None,
        }

    def _run_limits_payload(self, rec: _RunRecord) -> dict[str, JsonValue]:
        return {
            "schema_version": "synth.limits-progress.v1",
            "run_id": rec.run_id,
            "revision": rec.limits_revision,
            "as_of_ms": 1795802400000,
            "status": rec.state,
            "primary": [],
            "secondary": [],
            "actors": [],
            "allocation": {},
            "history": [],
            "funding": {
                "held": cents_to_money_str(rec.maximum_charge_cents),
                "settled": "0.00",
                "currency": "USD",
            },
        }

    def _post_create_run(self, body: dict[str, JsonValue]) -> httpx.Response:
        limits_raw = body.get("limits")
        limits = limits_raw if isinstance(limits_raw, dict) else {}
        mc_raw = limits.get("maximum_charge")
        mc = mc_raw if isinstance(mc_raw, dict) else {"amount": "5.00", "currency": "USD"}
        charge_cents = money_str_to_cents(str(mc.get("amount", "5.00")))
        wt_raw = limits.get("wall_time_seconds")
        pw_raw = body.get("parallel_workers")
        menu_rev = body.get("launch_menu_revision")
        ra_raw = body.get("role_assignments")
        role_assignments = ra_raw if isinstance(ra_raw, dict) else {}

        if role_assignments and menu_rev is None:
            return self._json_response(
                422,
                make_problem(
                    422, "launch_menu_revision_required", "role_assignments requires menu revision."
                ),
            )
        if menu_rev is not None:
            menu = self._hosted.get_menu(str(menu_rev))
            if menu is None:
                return self._json_response(
                    409,
                    make_problem(
                        409, "launch_menu_revision_stale", "launch_menu_revision is stale."
                    ),
                )
            refusals = self._hosted.evaluate_role_assignments(menu, role_assignments)
            if refusals:
                return self._json_response(
                    422, make_problem(422, str(refusals[0]["code"]), "Role assignment refused.")
                )
        if charge_cents < 50:
            return self._json_response(
                422, make_problem(422, "maximum_charge_too_small", "Minimum charge is $0.50.")
            )
        rec = _RunRecord(
            run_id=self._next_id("run"),
            thread_id=self._next_id("thr"),
            state="queued",
            limits_revision=1,
            maximum_charge_cents=charge_cents,
            wall_time_seconds=wt_raw if isinstance(wt_raw, int) else 3600,
            profile=str(body.get("profile", "swarms-v2")),
            domicile_requirement=str(body.get("domicile_requirement", "unrestricted")),
            parallel_workers=pw_raw if isinstance(pw_raw, int) else 1,
        )
        self._runs[rec.run_id] = rec
        return self._json_response(202, self._run_snapshot_payload(rec))

    def _post_run_control(self, run_id: str, action: str) -> httpx.Response:
        rec = self._runs.get(run_id)
        if rec is None:
            return self._json_response(
                404, make_problem(404, "run_not_found", f"Run {run_id} not found.")
            )
        if rec.state in _TERMINAL_RUN_STATES:
            return self._json_response(
                409, make_problem(409, "run_already_terminal", f"Run {run_id} is terminal.")
            )
        if action == "pause":
            if rec.state not in {"queued", "running", "pausing"}:
                return self._json_response(
                    409, make_problem(409, "run_not_pausable", f"Cannot pause {rec.state}.")
                )
            rec.state = "paused"
            reason = "pause_requested"
        elif action == "resume":
            if rec.state not in {"pausing", "paused"}:
                return self._json_response(
                    409, make_problem(409, "run_not_paused", f"Cannot resume {rec.state}.")
                )
            rec.state = "running"
            reason = "resume_requested"
        else:
            rec.state = "cancelling"
            return self._json_response(202, self._run_snapshot_payload(rec))
        return self._json_response(
            202,
            {
                "operation_id": self._next_id("ctl"),
                "resource_id": rec.run_id,
                "state": "accepted",
                "command": action,
                "reason": reason,
                "execution_closed": rec.execution_closed,
                "accounting_final": rec.accounting_final,
                "cleanup_complete": rec.cleanup_complete,
            },
        )

    def _post_run_limits(self, run_id: str, body: dict[str, JsonValue]) -> httpx.Response:
        rec = self._runs.get(run_id)
        if rec is None:
            return self._json_response(
                404, make_problem(404, "run_not_found", f"Run {run_id} not found.")
            )
        if rec.state in _TERMINAL_RUN_STATES:
            return self._json_response(
                409, make_problem(409, "run_already_terminal", "Cannot update terminal limits.")
            )
        if body.get("expected_revision") != rec.limits_revision:
            return self._json_response(
                409, make_problem(409, "limits_revision_conflict", "Stale expected_revision.")
            )
        limits_raw = body.get("limits")
        limits = limits_raw if isinstance(limits_raw, dict) else {}
        mc_raw = limits.get("maximum_charge")
        if isinstance(mc_raw, dict):
            new_charge = money_str_to_cents(str(mc_raw["amount"]))
            if new_charge < rec.maximum_charge_cents:
                return self._json_response(
                    422, make_problem(422, "limits_cannot_decrease", "Monotonic charge.")
                )
            rec.maximum_charge_cents = new_charge
        if body.get("resume") is True and rec.state in {"pausing", "paused"}:
            rec.state = "running"
        rec.limits_revision += 1
        return self._json_response(200, self._run_limits_payload(rec))

    def _post_run_message(self, run_id: str, body: dict[str, JsonValue]) -> httpx.Response:
        rec = self._runs.get(run_id)
        if rec is None:
            return self._json_response(
                404, make_problem(404, "run_not_found", f"Run {run_id} not found.")
            )
        if rec.state in _TERMINAL_RUN_STATES:
            return self._json_response(
                409, make_problem(409, "run_already_terminal", "Terminal run rejects messages.")
            )
        text = str(body.get("text", "")).strip()
        if not text:
            return self._json_response(
                422, make_problem(422, "message_text_empty", "Message text required.")
            )
        msg: dict[str, JsonValue] = {
            "message_id": self._next_id("msg"),
            "run_id": rec.run_id,
            "thread_id": rec.thread_id,
            "author": "customer",
            "kind": str(body.get("kind", "guidance")),
            "sequence": len(rec.messages) + 1,
            "body": {"kind": "available", "text": text},
            "state": {"kind": "queued"},
            "created_at": _FIXED_TS,
            "updated_at": _FIXED_TS,
        }
        rec.messages.append(msg)
        return self._json_response(201, msg)


class SdkPbtHarness:
    """Pairs a real SynthClient with either an in-process MockTransport or live local-slot base URL."""

    def __init__(self, config: SdkPbtConfig) -> None:
        self.config = config
        self.backend: InProcessSdkBackend | None = None
        self._http: httpx.AsyncClient | None = None
        if config.base_url is not None and config.base_url != "":
            self.client = SynthClient(
                config.api_key or "fixture", base_url=config.base_url, retries=0
            )
        else:
            self.backend = InProcessSdkBackend()
            self._http = httpx.AsyncClient(
                transport=httpx.MockTransport(self.backend.handle_request)
            )
            self.client = SynthClient(
                "fixture",
                base_url="https://pbt.synth.invalid",
                http_client=self._http,
                retries=0,
            )

    @property
    def is_live(self) -> bool:
        return self.backend is None

    def raw_conditional_get(self, path: str, *, if_none_match: str) -> httpx.Response:
        """Issue a raw conditional GET with If-None-Match to verify ETag 304 stability."""
        if self.backend is not None:
            req = httpx.Request(
                "GET", f"https://pbt.synth.invalid{path}", headers={"If-None-Match": if_none_match}
            )
            return self.backend.handle_request(req)
        assert self.config.base_url is not None
        hdrs = {"If-None-Match": if_none_match, "Accept": "application/json"}
        if self.config.api_key is not None and self.config.api_key != "":
            hdrs["Authorization"] = f"Bearer {self.config.api_key}"
        return httpx.get(f"{self.config.base_url.rstrip('/')}{path}", headers=hdrs, timeout=15.0)

    def close(self) -> None:
        self.client.close()
        self._http = None
