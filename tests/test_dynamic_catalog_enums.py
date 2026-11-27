"""SYN-6038 & SYN-6059: forward-compatible catalog enum decoding and pinned catalog codegen."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_cli import Api, example, invoke

from synth_ai import AsyncSynthClient, OptimizerPin, topology_name
from synth_ai._cli_view import promo_redemption
from synth_ai._generated.billing_plans import (
    AUTO_RELOAD_MAX_AMOUNT_CENTS,
    AUTO_RELOAD_MIN_THRESHOLD_CENTS,
    CREDIT_PACK_IDS,
    CUSTOM_CREDIT_MAX_CENTS,
    CUSTOM_CREDIT_MIN_CENTS,
    SWITCHABLE_PLANS,
)
from synth_ai._generated.models import (
    CheckoutRequest,
    FamilyList,
    PromoRedemption,
    ResolvedOptimizer,
    SandboxProfile,
    SubscriptionChangeRequest,
    TopologyCatalogEntry,
)
from synth_ai._generated.topologies import (
    DEFAULT_RUN_PROFILE,
    EXPERIMENTAL_TOPOLOGIES,
    PROFILE_REQUIRES_RESEARCH_TASK,
    PUBLIC_TOPOLOGIES,
    TOPOLOGY_DISPLAY_NAMES,
    TOPOLOGY_PARAM_BOUNDS,
    TOPOLOGY_PARAM_NAMES,
)
from synth_ai.cli import EXIT_OK
from synth_ai.mcp.server import call_tool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(contract, monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("SYNTH_STATE_DIR", str(tmp_path / "state"))
    return Api(contract)


def test_generated_catalog_matches_pinned_json():
    topologies = json.loads((ROOT / "contracts/catalog/topologies.json").read_text())
    billing = json.loads((ROOT / "contracts/catalog/billing_plans.json").read_text())

    assert (
        tuple(item["id"] for item in topologies["topologies"] if item["visibility"] == "public")
        == PUBLIC_TOPOLOGIES
    )
    assert {
        item["id"]: item["display_name"] for item in topologies["topologies"]
    } == TOPOLOGY_DISPLAY_NAMES
    assert (
        frozenset(
            item["id"] for item in topologies["topologies"] if item["label"] == "experimental"
        )
        == EXPERIMENTAL_TOPOLOGIES
    )
    assert set(TOPOLOGY_PARAM_NAMES["fork_flush"]) == {"branches", "rounds", "exploit_bps"}
    assert TOPOLOGY_PARAM_BOUNDS["branches"] == (2, 32, 4)
    assert DEFAULT_RUN_PROFILE == "swarms-v2"
    assert PROFILE_REQUIRES_RESEARCH_TASK == {"fork-flush": False, "swarms-v2": False}

    assert tuple(pack["pack_id"] for pack in billing["credit_packs"]) == CREDIT_PACK_IDS
    assert tuple(plan["id"] for plan in billing["plans"] if plan["switchable"]) == SWITCHABLE_PLANS
    assert billing["custom_credit"]["min_cents"] == CUSTOM_CREDIT_MIN_CENTS
    assert billing["custom_credit"]["max_cents"] == CUSTOM_CREDIT_MAX_CENTS
    assert billing["auto_reload"]["min_threshold_cents"] == AUTO_RELOAD_MIN_THRESHOLD_CENTS
    assert billing["auto_reload"]["max_amount_cents"] == AUTO_RELOAD_MAX_AMOUNT_CENTS


def test_unknown_future_catalog_values_decode_without_contract_error():
    entry = TopologyCatalogEntry.model_validate(
        {
            "topology": "tree_search",
            "version": 1,
            "label": "default",
            "auto": False,
            "availability": "served",
        }
    )
    assert entry.topology == "tree_search"
    assert topology_name(entry) == "Tree Search"

    resolved = ResolvedOptimizer.model_validate(
        {
            "topology": "tree_search",
            "version": 1,
            "params": {},
            "reason": "explicit_pin",
            "definition_digest": "sha256:" + "a" * 64,
        }
    )
    assert resolved.topology == "tree_search"

    pin = OptimizerPin.model_validate({"topology": "tree_search", "version": 3})
    assert pin.topology == "tree_search"

    sandbox = SandboxProfile.model_validate(
        {
            "actor_id": "worker",
            "profile_id": "8vcpu-32gib-h200-us",
            "profile_digest": "sha256:" + "b" * 64,
            "size_id": "8vcpu-32gib",
            "domicile": "us",
            "gpu_kind": "h200",
            "gpu_count": 1,
        }
    )
    assert sandbox.gpu_kind == "h200"

    checkout = CheckoutRequest.model_validate({"pack_id": "credits.500"})
    assert checkout.pack_id == "credits.500"

    change = SubscriptionChangeRequest.model_validate({"plan_id": "enterprise.monthly"})
    assert change.plan_id == "enterprise.monthly"

    redemption = PromoRedemption.model_validate(
        {
            "redemption_id": "prd_AAAAAAAAAAAAAAAAAAAAAA",
            "campaign_id": "camp_future",
            "org_id": "org_1",
            "redeemed_at": "2026-11-27T18:00:00Z",
            "grant_ids": [],
            "override_ids": [],
            "trial": {
                "trial_id": "trl_1",
                "plan_id": "enterprise.monthly",
                "starts_at": "2026-11-27T18:00:00Z",
                "ends_at": "2026-12-27T18:00:00Z",
            },
            "replayed": False,
        }
    )
    assert redemption.trial is not None
    assert redemption.trial.plan_id == "enterprise.monthly"
    assert "enterprise.monthly" in promo_redemption(redemption)


@pytest.mark.parametrize("bad", ["", "   ", "bad\nvalue", 123])
def test_open_catalog_enums_still_refuse_malformed_values(bad):
    with pytest.raises(ValidationError):
        OptimizerPin.model_validate({"topology": bad, "version": 1})
    with pytest.raises(ValidationError):
        CheckoutRequest.model_validate({"pack_id": bad})
    with pytest.raises(ValidationError):
        SubscriptionChangeRequest.model_validate({"plan_id": bad})


async def test_unknown_server_catalog_responses_and_requests_work_over_sdk_and_mcp(contract):
    _, fam_raw = example(contract, "listFamilies")
    fam_raw["topologies"] = [
        {
            "topology": "tree_search",
            "version": 1,
            "label": "experimental",
            "auto": False,
            "availability": "served",
        }
    ]
    _, snap_raw = example(contract, "getRun")
    snap_raw["optimizer"] = {
        "topology": "tree_search",
        "version": 1,
        "params": {},
        "reason": "explicit_pin",
        "definition_digest": "sha256:" + "c" * 64,
    }
    _, checkout_raw = example(contract, "createCheckout")
    _, sub_raw = example(contract, "changeSubscriptionPlan")
    sub_raw["plan_id"] = "enterprise.monthly"

    sent = []

    def handle(request):
        if request.content:
            sent.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/families":
            return httpx.Response(200, json=fam_raw)
        if request.url.path == "/v1/runs":
            return httpx.Response(202, json=snap_raw)
        if request.url.path == "/v1/billing/checkout":
            return httpx.Response(201, json=checkout_raw)
        if request.url.path == "/v1/billing/subscription/change":
            return httpx.Response(200, json=sub_raw)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            fams = await client.swarms.families()
            assert isinstance(fams, FamilyList)
            assert fams.topologies is not None
            assert fams.topologies[0].topology == "tree_search"

            run = await client.swarms.launch(
                "task", budget="1.00", optimizer=OptimizerPin(topology="tree_search", version=1)
            )
            assert run.optimizer is not None
            assert run.optimizer.topology == "tree_search"

            await client.billing.checkout(pack_id="credits.500", idempotency_key="ck-future")
            sub = await client.billing.change_subscription_plan(
                "enterprise.monthly", idempotency_key="sp-future"
            )
            assert sub.plan_id == "enterprise.monthly"

            # MCP accepts newly graduated server catalog values cleanly as well.
            await call_tool(
                client,
                "swarms_run_launch",
                {
                    "body": {
                        "message": "task",
                        "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
                        "optimizer": {"topology": "tree_search", "version": 1},
                    },
                    "idempotency_key": "mcp-future-topo",
                },
            )
            await call_tool(
                client,
                "account_billing_checkout_create",
                {"body": {"pack_id": "credits.500"}, "idempotency_key": "mcp-future-pack"},
            )
            await call_tool(
                client,
                "account_billing_subscription_change",
                {"body": {"plan_id": "enterprise.monthly"}, "idempotency_key": "mcp-future-plan"},
            )


def test_unknown_server_catalog_requests_work_over_cli(contract, api, capsys):
    _, snap_raw = example(contract, "getRun")
    snap_raw["optimizer"] = {
        "topology": "tree_search",
        "version": 1,
        "params": {},
        "reason": "explicit_pin",
        "definition_digest": "sha256:" + "c" * 64,
    }
    _, checkout_raw = example(contract, "createCheckout")
    _, sub_raw = example(contract, "changeSubscriptionPlan")
    sub_raw["plan_id"] = "enterprise.monthly"
    api.overrides["createCheckout"] = (201, checkout_raw)
    api.overrides["changeSubscriptionPlan"] = (200, sub_raw)
    api.overrides["createRun"] = (202, snap_raw)
    for argv in (
        ("billing", "checkout", "--pack", "credits.500", "--json"),
        ("billing", "change-plan", "--plan", "enterprise.monthly", "--json"),
        ("run", "Goal", "--budget", "1.00", "--optimizer", "tree_search@1", "--json"),
    ):
        code, _, err = invoke(api, capsys, *argv)
        assert code == EXIT_OK, err
