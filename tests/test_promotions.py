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
    assert observed == ["same", "same", "same"]
    assert json.loads(capsys.readouterr().out)["campaign_id"] == response["campaign_id"]
