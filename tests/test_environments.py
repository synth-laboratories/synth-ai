"""Named targets preserve credential host isolation and refuse ambiguous selection."""

import httpx
import pytest

from synth_ai import AsyncSynthClient, Environment, SynthClient, cli
from synth_ai.errors import InvalidRequestError, SwarmsAPIError


async def test_sdk_selects_environment_without_global_origin_override(monkeypatch):
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://production.example")
    monkeypatch.setenv("SYNTH_STAGING_API_BASE_URL", "https://staging.example")
    seen = []

    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", environment=Environment.STAGING, http_client=http
        ) as client:
            with pytest.raises(SwarmsAPIError):
                await client.account.get()
    assert seen == ["https://staging.example/v1/account"]


def test_cli_selects_environment():
    arguments = cli.parser().parse_args(["--env", "staging", "runs", "list", "--json"])
    assert arguments.environment == "staging"


def test_sync_sdk_exposes_named_target():
    with SynthClient("fixture", environment=Environment.PROD):
        pass


def test_profile_rejects_conflicting_and_missing_origins(monkeypatch):
    monkeypatch.delenv("SYNTH_LOCAL_API_BASE_URL", raising=False)
    with pytest.raises(InvalidRequestError, match="environment_profile_unconfigured"):
        AsyncSynthClient("fixture", environment=Environment.LOCAL)
    with pytest.raises(InvalidRequestError, match="environment_profile_conflict"):
        AsyncSynthClient(
            "fixture", environment=Environment.STAGING, base_url="https://production.example"
        )


@pytest.mark.parametrize(
    "origin",
    ["", "http://staging.example", "https://u:p@staging.example", "https://staging.example/path"],
)
def test_staging_refuses_invalid_origin(monkeypatch, origin):
    monkeypatch.setenv("SYNTH_STAGING_API_BASE_URL", origin)
    with pytest.raises(InvalidRequestError, match="environment_origin_invalid"):
        AsyncSynthClient("fixture", environment=Environment.STAGING)
