"""Explicit account consent is advertised only by configured account deployments."""

import asyncio

import httpx
import pytest
from synth_ai.mcp.hosted import AUTHORIZATION_SCOPES, create_http_app


def metadata(*, account):
    app = create_http_app(
        object(),
        resource_url="https://sdk.fixture.invalid/mcp",
        issuer_url="https://issuer.fixture.invalid/",
        enabled_tools=frozenset({"account_usage_snapshot"}) if account else frozenset(),
        account_capability_provider=(lambda: None) if account else None,
        account_oauth_scopes_supported=account,
    )

    async def read():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://sdk.fixture.invalid"
        ) as client:
            response = await client.get("/.well-known/oauth-protected-resource/mcp")
            assert response.status_code == 200
            return response.json()

    return asyncio.run(read())


def test_account_permissions_are_explicit_and_opt_in():
    default = metadata(account=False)
    assert default["scopes_supported"] == list(AUTHORIZATION_SCOPES)
    account = metadata(account=True)
    assert set(account["scopes_supported"]) == set(AUTHORIZATION_SCOPES) | {
        "account:billing:read",
        "account:usage:read",
    }


def test_account_scope_advertisement_refuses_unconfigured_tool():
    with pytest.raises(ValueError):
        create_http_app(
            object(),
            resource_url="https://sdk.fixture.invalid/mcp",
            issuer_url="https://issuer.fixture.invalid/",
            enabled_tools=frozenset(),
            account_oauth_scopes_supported=True,
        )
