"""Supplier-neutral hosted options menu builder & preflight evaluator for SDK PBT (SYN-6050)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal

from pydantic import JsonValue

__all__ = [
    "HostedOptionsEngine",
    "cents_to_money_str",
    "money_str_to_cents",
]


def cents_to_money_str(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    return f"{sign}{whole}.{frac:02d}"


def money_str_to_cents(amount: str) -> int:
    dec = Decimal(amount)
    return int((dec * 100).to_integral_exact())


def _money(cents: int) -> dict[str, JsonValue]:
    return {"amount": cents_to_money_str(cents), "currency": "USD"}


def _stable_json_bytes(obj: JsonValue) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


class HostedOptionsEngine:
    """Deterministic supplier-neutral engine for /v1/launch-choices and /v1/runs/preflight."""

    def __init__(self) -> None:
        self._known_menus: dict[str, dict[str, JsonValue]] = {}

    def get_menu(self, menu_revision: str) -> dict[str, JsonValue] | None:
        return self._known_menus.get(menu_revision)

    def build_launch_menu(
        self,
        *,
        profile: str,
        domicile: str,
        parallel_workers: int,
        wall_time_seconds: int,
        maximum_charge_cents: int,
    ) -> dict[str, JsonValue]:
        inf_specs: list[tuple[str, str, int, int, int, str, int]] = [
            ("luna-low", "low", 50, 125, 1000, "0.125", 50),
            ("luna-high", "high", 200, 250, 1500, "0.25", 200),
            ("luna-xhigh", "xhigh", 1000, 500, 2500, "0.50", 1000),
        ]
        sbx_specs: list[tuple[str, str, str, int, int, int, int, str]] = [
            (
                "2vcpu-2gib-us",
                "2vcpu-2gib",
                "us",
                2000,
                2147483648,
                10737418240,
                10,
                "sbo1_0000000000000001_" + "a" * 64,
            ),
            (
                "4vcpu-8gib-us",
                "4vcpu-8gib",
                "us",
                4000,
                8589934592,
                21474836480,
                40,
                "sbo1_0000000000000002_" + "b" * 64,
            ),
            (
                "8vcpu-16gib-us",
                "8vcpu-16gib",
                "us",
                8000,
                17179869184,
                42949672960,
                120,
                "sbo1_0000000000000003_" + "c" * 64,
            ),
            (
                "4vcpu-8gib-eu",
                "4vcpu-8gib",
                "eu",
                4000,
                8589934592,
                21474836480,
                45,
                "sbo1_0000000000000004_" + "d" * 64,
            ),
        ]

        inf_admitted: list[JsonValue] = []
        exclusions: list[JsonValue] = []
        for idx, (pid, effort, cap_c, in_c, out_c, cached_s, min_b) in enumerate(
            inf_specs, start=1
        ):
            if maximum_charge_cents >= min_b:
                inf_admitted.append(
                    {
                        "profile_id": pid,
                        "model": "gpt-6-luna",
                        "effort": effort,
                        "profile_digest": f"sha256:{str(idx) * 64}",
                        "context_window": 1050000,
                        "maximum_output_tokens": 128000,
                        "per_call_spend_cap": _money(cap_c),
                        "input_price_per_million_tokens": _money(in_c),
                        "cached_input_price_per_million_tokens": {
                            "amount": cached_s,
                            "currency": "USD",
                        },
                        "output_price_per_million_tokens": _money(out_c),
                    }
                )
            else:
                exclusions.append(
                    {
                        "kind": "inference",
                        "profile_id": pid,
                        "cause": "affordability",
                        "code": "launch_choice_unaffordable",
                    }
                )

        hours = max(1, (wall_time_seconds + 3599) // 3600)
        sbx_admitted: list[JsonValue] = []
        for pid, size_id, dom, cpu, mem, disk, hr_c, opt_id in sbx_specs:
            if domicile == "us_only" and dom != "us":
                exclusions.append(
                    {
                        "kind": "sandbox",
                        "profile_id": pid,
                        "cause": "locality",
                        "code": "launch_choice_domicile_excluded",
                    }
                )
                continue
            if hr_c * hours * max(1, parallel_workers) > maximum_charge_cents:
                exclusions.append(
                    {
                        "kind": "sandbox",
                        "profile_id": pid,
                        "cause": "affordability",
                        "code": "launch_choice_unaffordable",
                    }
                )
                continue
            sbx_admitted.append(
                {
                    "profile_id": pid,
                    "size_id": size_id,
                    "domicile": dom,
                    "cpu_millis": cpu,
                    "memory_bytes": mem,
                    "disk_bytes": disk,
                    "hourly_price": _money(hr_c),
                    "option_required": False,
                    "options": [{"option_id": opt_id, "hourly_price": _money(hr_c)}],
                }
            )

        first_inf = (
            inf_admitted[0]["profile_id"]
            if inf_admitted and isinstance(inf_admitted[0], dict)
            else None
        )
        first_sbx = (
            sbx_admitted[0]["profile_id"]
            if sbx_admitted and isinstance(sbx_admitted[0], dict)
            else None
        )
        body_no_rev: dict[str, JsonValue] = {
            "run_profile": profile,
            "domicile_requirement": domicile,
            "affordability": {
                "status": "evaluated",
                "maximum_charge": _money(maximum_charge_cents),
                "wall_time_seconds": wall_time_seconds,
            },
            "roles": [
                {
                    "actor_id": "planner",
                    "role": "planner",
                    "default_inference_profile": first_inf,
                    "default_sandbox_profile": first_sbx,
                    "planner_decides_allowed": False,
                },
                {
                    "actor_id": "worker",
                    "role": "worker",
                    "default_inference_profile": first_inf,
                    "default_sandbox_profile": None,
                    "planner_decides_allowed": True,
                },
            ],
            "inference_profiles": inf_admitted,
            "sandbox_profiles": sbx_admitted,
            "exclusions": exclusions,
        }
        menu_revision = "lcm1_" + hashlib.sha256(_stable_json_bytes(body_no_rev)).hexdigest()
        payload: dict[str, JsonValue] = {"menu_revision": menu_revision, **body_no_rev}
        self._known_menus[menu_revision] = payload
        return payload

    def evaluate_role_assignments(
        self, menu: Mapping[str, JsonValue], role_assignments: Mapping[str, JsonValue]
    ) -> list[dict[str, JsonValue]]:
        inf_list = menu.get("inference_profiles")
        sbx_list = menu.get("sandbox_profiles")
        valid_inf = {
            str(item["profile_id"])
            for item in (inf_list if isinstance(inf_list, list) else [])
            if isinstance(item, dict)
        }
        valid_sbx: dict[str, set[str]] = {}
        for item in sbx_list if isinstance(sbx_list, list) else []:
            if isinstance(item, dict):
                opts = item.get("options")
                valid_sbx[str(item["profile_id"])] = {
                    str(o["option_id"])
                    for o in (opts if isinstance(opts, list) else [])
                    if isinstance(o, dict)
                }
        refusals: list[dict[str, JsonValue]] = []
        for actor_id, assign in role_assignments.items():
            if not isinstance(assign, dict):
                refusals.append(
                    {"check": "role_assignments", "code": "role_assignment_invalid", "status": 422}
                )
                continue
            inf = assign.get("inference_profile")
            if isinstance(inf, str) and inf not in valid_inf:
                refusals.append(
                    {
                        "check": "role_assignments",
                        "code": "launch_choice_inference_excluded",
                        "status": 422,
                        "detail": f"{actor_id} inference {inf} excluded",
                    }
                )
            sbx = assign.get("sandbox_profile")
            if isinstance(sbx, str) and sbx not in valid_sbx:
                refusals.append(
                    {
                        "check": "role_assignments",
                        "code": "launch_choice_sandbox_excluded",
                        "status": 422,
                        "detail": f"{actor_id} sandbox {sbx} excluded",
                    }
                )
            opt = assign.get("sandbox_option")
            if isinstance(opt, str) and (
                not isinstance(sbx, str) or sbx not in valid_sbx or opt not in valid_sbx[sbx]
            ):
                refusals.append(
                    {
                        "check": "role_assignments",
                        "code": "launch_choice_option_excluded",
                        "status": 422,
                        "detail": f"{actor_id} option {opt} excluded",
                    }
                )
        return refusals

    def evaluate_preflight(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        limits_raw = body.get("limits")
        limits = limits_raw if isinstance(limits_raw, dict) else {}
        mc_raw = limits.get("maximum_charge")
        mc = mc_raw if isinstance(mc_raw, dict) else {"amount": "5.00", "currency": "USD"}
        charge_cents = money_str_to_cents(str(mc.get("amount", "5.00")))
        menu_rev = body.get("launch_menu_revision")
        ra_raw = body.get("role_assignments")
        role_assignments = ra_raw if isinstance(ra_raw, dict) else {}

        checks: list[JsonValue] = [{"name": "request_schema", "status": "pass"}]
        refusals: list[JsonValue] = []
        if charge_cents < 50:
            checks.append(
                {"name": "wallet_balance", "status": "fail", "code": "maximum_charge_too_small"}
            )
            refusals.append(
                {"check": "wallet_balance", "code": "maximum_charge_too_small", "status": 422}
            )
        else:
            checks.append({"name": "wallet_balance", "status": "pass"})

        if role_assignments and menu_rev is None:
            checks.append(
                {
                    "name": "role_assignments",
                    "status": "fail",
                    "code": "launch_menu_revision_required",
                }
            )
            refusals.append(
                {
                    "check": "role_assignments",
                    "code": "launch_menu_revision_required",
                    "status": 422,
                }
            )
        elif menu_rev is not None:
            menu = self._known_menus.get(str(menu_rev))
            if menu is None:
                checks.append(
                    {
                        "name": "role_assignments",
                        "status": "fail",
                        "code": "launch_menu_revision_stale",
                    }
                )
                refusals.append(
                    {
                        "check": "role_assignments",
                        "code": "launch_menu_revision_stale",
                        "status": 409,
                    }
                )
            else:
                ra_refusals = self.evaluate_role_assignments(menu, role_assignments)
                if ra_refusals:
                    checks.append(
                        {
                            "name": "role_assignments",
                            "status": "fail",
                            "code": ra_refusals[0]["code"],
                        }
                    )
                    refusals.extend(ra_refusals)
                else:
                    checks.append({"name": "role_assignments", "status": "pass"})

        return {
            "schema_version": "synth.run-preflight.v1",
            "admissible": len(refusals) == 0,
            "checks": checks,
            "refusals": refusals,
        }
