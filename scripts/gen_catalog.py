#!/usr/bin/env python3
"""Generate topology and billing catalog constants from Core's pinned JSON catalogs."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]


class PinnedFile(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CatalogFiles(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    billing_plans: PinnedFile
    topologies: PinnedFile


class CatalogProvenance(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    repository: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: CatalogFiles


class EnvMatrix(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    local: bool
    staging: bool
    production: bool


class ParamSpec(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    type: Literal["integer"]
    min: int
    max: int
    default: int
    choice_key: str


class TopologyEntry(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    id: str
    version: int = Field(ge=1)
    display_name: str
    visibility: Literal["public", "research"]
    label: str
    auto: bool
    family_id: str | None
    family_version: str | None
    availability: EnvMatrix
    params: dict[str, ParamSpec]


class RunProfileEntry(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    id: str
    revision: str
    name: str
    description: str
    topology_id: str
    requires_research_task: bool
    program_id: str
    program_version: str
    family_id: str
    method_release: str | None
    default: bool
    deferred: bool
    published: EnvMatrix


class TopologiesCatalog(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: Literal["synth.topologies.v1"] = Field(alias="schema")
    environments: list[str]
    topologies: list[TopologyEntry]
    run_profiles: list[RunProfileEntry]


class CurrencyUnits(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    microcents_per_usd: int = Field(gt=0)
    microcents_per_cent: int = Field(gt=0)
    microcents_per_micro: int = Field(gt=0)


class BillingConstants(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    unlimited: int
    minimum_swarm_concurrency: int = Field(gt=0)
    max_override_lifetime_days: int = Field(gt=0)
    promo_trial_max_days: int = Field(gt=0)
    spend_resources: list[str]


class PlanLimits(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    concurrent_runs: int
    concurrent_rollouts: int
    daily_spend_usd: int | None
    weekly_spend_usd: int | None


class PlanEntry(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    id: str
    name: str
    kind: Literal["free", "granted", "recurring", "credit_pack", "internal"]
    price_cents: int = Field(ge=0)
    interval: str | None
    weekly_included_microcents: int = Field(ge=0)
    included_usage_limit_microcents: int = Field(ge=0)
    stripe_lookup_key: str | None
    purchasable: bool
    switchable: bool
    trial_eligible: bool
    limits: PlanLimits
    rhodes_job_caps: dict[str, int]


class PaygTier(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    daily_spend_usd: int = Field(gt=0)
    weekly_spend_usd: int = Field(gt=0)
    concurrent_runs: int = Field(gt=0)
    concurrent_rollouts: int = Field(gt=0)


class CreditPackEntry(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    pack_id: str
    name: str
    price_cents: int = Field(gt=0)


class CustomCreditBounds(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    min_cents: int = Field(gt=0)
    max_cents: int = Field(gt=0)


class AutoReloadBounds(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    daily_cap: int = Field(gt=0)
    monthly_ceiling_cents: int = Field(gt=0)
    min_threshold_cents: int = Field(gt=0)
    max_threshold_cents: int = Field(gt=0)
    min_amount_cents: int = Field(gt=0)
    max_amount_cents: int = Field(gt=0)


class SpendAlertBounds(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    max_monthly_microcents: int = Field(gt=0)
    thresholds_percent: list[int]


class PlanRequestBounds(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    body_bytes_max: int = Field(gt=0)
    use_case_bytes_max: int = Field(gt=0)
    links_max: int = Field(gt=0)
    url_bytes_max: int = Field(gt=0)
    note_bytes_max: int = Field(gt=0)
    list_limit: int = Field(gt=0)
    denial_cooldown_days: int = Field(gt=0)


class BillingPlansCatalog(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: Literal["synth.billing-plans.v1"] = Field(alias="schema")
    policy_version: str
    currency_units: CurrencyUnits
    constants: BillingConstants
    plans: list[PlanEntry]
    payg_tier: PaygTier
    credit_packs: list[CreditPackEntry]
    custom_credit: CustomCreditBounds
    auto_reload: AutoReloadBounds
    spend_alerts: SpendAlertBounds
    plan_requests: PlanRequestBounds


def _format_python(path: Path, source: str) -> str:
    fixed = subprocess.run(
        [
            "ruff",
            "check",
            "--select",
            "I,RUF036,E713",
            "--fix",
            "--stdin-filename",
            str(path),
            "-",
        ],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return subprocess.run(
        ["ruff", "format", "--stdin-filename", str(path), "-"],
        input=fixed,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _read_verified(root: Path, pinned: PinnedFile) -> bytes:
    raw = (root / pinned.path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != pinned.sha256:
        raise ValueError(f"{pinned.path} differs from CATALOG_SOURCE.toml sha256")
    return raw


def _render_topologies(source: CatalogProvenance, catalog: TopologiesCatalog) -> str:
    display_names = {entry.id: entry.display_name for entry in catalog.topologies}
    public_topologies = tuple(
        entry.id for entry in catalog.topologies if entry.visibility == "public"
    )
    experimental = sorted(entry.id for entry in catalog.topologies if entry.label == "experimental")
    param_names = {entry.id: tuple(entry.params) for entry in catalog.topologies if entry.params}
    param_bounds = {
        param_name: (spec.min, spec.max, spec.default)
        for entry in catalog.topologies
        for param_name, spec in entry.params.items()
    }
    defaults = [profile.id for profile in catalog.run_profiles if profile.default]
    if len(defaults) != 1:
        raise ValueError("topologies.json must declare exactly one default run profile")
    profile_requirements = {
        profile.id: profile.requires_research_task
        for profile in sorted(catalog.run_profiles, key=lambda item: item.id)
    }
    lines = [
        "# Generated by scripts/gen_catalog.py; do not edit.",
        f"# Core source: {source.commit}:{source.files.topologies.path}",
        "from __future__ import annotations",
        "",
        f"CATALOG_COMMIT = {source.commit!r}",
        f"TOPOLOGIES_SHA256 = {source.files.topologies.sha256!r}",
        f"PUBLIC_TOPOLOGIES: tuple[str, ...] = {public_topologies!r}",
        f"TOPOLOGY_DISPLAY_NAMES: dict[str, str] = {display_names!r}",
        f"EXPERIMENTAL_TOPOLOGIES: frozenset[str] = frozenset({experimental!r})",
        f"TOPOLOGY_PARAM_NAMES: dict[str, tuple[str, ...]] = {param_names!r}",
        f"TOPOLOGY_PARAM_BOUNDS: dict[str, tuple[int, int, int]] = {param_bounds!r}",
        f"DEFAULT_RUN_PROFILE = {defaults[0]!r}",
        f"PROFILE_REQUIRES_RESEARCH_TASK: dict[str, bool] = {profile_requirements!r}",
        "",
    ]
    return "\n".join(lines)


def _render_billing_plans(source: CatalogProvenance, catalog: BillingPlansCatalog) -> str:
    switchable = tuple(plan.id for plan in catalog.plans if plan.switchable)
    pack_ids = tuple(pack.pack_id for pack in catalog.credit_packs)
    lines = [
        "# Generated by scripts/gen_catalog.py; do not edit.",
        f"# Core source: {source.commit}:{source.files.billing_plans.path}",
        "from __future__ import annotations",
        "",
        f"CATALOG_COMMIT = {source.commit!r}",
        f"BILLING_PLANS_SHA256 = {source.files.billing_plans.sha256!r}",
        f"POLICY_VERSION = {catalog.policy_version!r}",
        f"SWITCHABLE_PLANS: tuple[str, ...] = {switchable!r}",
        f"CREDIT_PACK_IDS: tuple[str, ...] = {pack_ids!r}",
        f"CUSTOM_CREDIT_MIN_CENTS = {catalog.custom_credit.min_cents}",
        f"CUSTOM_CREDIT_MAX_CENTS = {catalog.custom_credit.max_cents}",
        f"AUTO_RELOAD_DAILY_CAP = {catalog.auto_reload.daily_cap}",
        f"AUTO_RELOAD_MONTHLY_CEILING_CENTS = {catalog.auto_reload.monthly_ceiling_cents}",
        f"AUTO_RELOAD_MIN_THRESHOLD_CENTS = {catalog.auto_reload.min_threshold_cents}",
        f"AUTO_RELOAD_MAX_THRESHOLD_CENTS = {catalog.auto_reload.max_threshold_cents}",
        f"AUTO_RELOAD_MIN_AMOUNT_CENTS = {catalog.auto_reload.min_amount_cents}",
        f"AUTO_RELOAD_MAX_AMOUNT_CENTS = {catalog.auto_reload.max_amount_cents}",
        f"SPEND_ALERT_MAX_MONTHLY_MICROCENTS = {catalog.spend_alerts.max_monthly_microcents}",
        f"SPEND_ALERT_THRESHOLDS_PERCENT: tuple[int, ...] = {tuple(catalog.spend_alerts.thresholds_percent)!r}",
        f"PAYG_DAILY_SPEND_USD = {catalog.payg_tier.daily_spend_usd}",
        f"PAYG_WEEKLY_SPEND_USD = {catalog.payg_tier.weekly_spend_usd}",
        "",
    ]
    return "\n".join(lines)


def generate(root: Path = ROOT) -> dict[Path, str]:
    source = CatalogProvenance.model_validate(
        tomllib.loads((root / "contracts/CATALOG_SOURCE.toml").read_text())
    )
    billing_raw = _read_verified(root, source.files.billing_plans)
    topologies_raw = _read_verified(root, source.files.topologies)
    billing = BillingPlansCatalog.model_validate_json(billing_raw)
    topologies = TopologiesCatalog.model_validate_json(topologies_raw)
    topologies_path = root / "synth_ai/_generated/topologies.py"
    billing_path = root / "synth_ai/_generated/billing_plans.py"
    return {
        topologies_path: _format_python(topologies_path, _render_topologies(source, topologies)),
        billing_path: _format_python(billing_path, _render_billing_plans(source, billing)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, expected in generate().items():
        if args.check:
            if not path.exists() or path.read_text() != expected:
                raise SystemExit(f"generated catalog drift: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)


if __name__ == "__main__":
    main()
