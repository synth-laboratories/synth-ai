"""Mock HTTP verifies the real public SDK/CLI/MCP delegation and mutation key."""

import json

import httpx
from test_cli import example

from synth_ai import AsyncSynthClient, cli
from synth_ai.mcp.server import call_tool


async def test_promotions_sdk_cli_mcp_share_contract_and_key(contract, capsys, monkeypatch):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    observed = []
    status, response = example(contract, "redeemPromo")

    def handle(request):
        assert request.method == "POST" and request.url.path == "/v1/promotions/redeem"
        assert json.loads(request.content) == {"code": "CODE"}
        observed.append(request.headers["Idempotency-Key"])
        return httpx.Response(status, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = await client.account.redeem_promo("CODE", idempotency_key="same")
            assert str(result.redemption_id) == response["redemption_id"]
            result = json.loads(
                await call_tool(
                    client,
                    "account_promo_redeem",
                    {"body": {"code": "CODE"}, "idempotency_key": "same"},
                )
            )
            assert result["grant_ids"] == response["grant_ids"]
        args = cli.parser().parse_args(
            [
                "--base-url",
                "https://example.test",
                "promo",
                "redeem",
                "CODE",
                "--idempotency-key",
                "same",
            ]
        )
        assert await cli.execute(args, http) == 0
        human = capsys.readouterr().out
        assert not human.lstrip().startswith("{")
        assert response["redemption_id"] in human
        assert response["campaign_id"] in human
        json_args = cli.parser().parse_args(
            [
                "--base-url",
                "https://example.test",
                "promo",
                "redeem",
                "CODE",
                "--idempotency-key",
                "same",
                "--json",
            ]
        )
        assert await cli.execute(json_args, http) == 0
        raw_json = capsys.readouterr().out
    assert observed == ["same", "same", "same", "same"]
    assert len(raw_json.strip().splitlines()) == 1
    assert json.loads(raw_json)["campaign_id"] == response["campaign_id"]


def test_promo_redemption_human_view_renders_overrides_and_trial():
    from synth_ai import _cli_view as view
    from synth_ai._generated.models import PromoRedemption

    redemption = PromoRedemption.model_validate(
        {
            "redemption_id": "prd_AAAAAAAAAAAAAAAAAAAAAA",
            "campaign_id": "camp_launch",
            "org_id": "org_1",
            "redeemed_at": "2026-11-27T18:00:00Z",
            "grant_ids": ["grt_1", "grt_2"],
            "override_ids": ["ovr_1"],
            "trial": {
                "trial_id": "trl_1",
                "plan_id": "standard.monthly",
                "starts_at": "2026-11-27T18:00:00Z",
                "ends_at": "2026-12-27T18:00:00Z",
            },
            "replayed": True,
        }
    )
    rendered = view.promo_redemption(redemption)
    assert "prd_AAAAAAAAAAAAAAAAAAAAAA" in rendered
    assert "camp_launch" in rendered
    assert "replayed" in rendered and "yes" in rendered
    assert "grt_1, grt_2" in rendered
    assert "ovr_1" in rendered
    assert "standard.monthly (trl_1, ends 2026-12-27T18:00:00Z)" in rendered
