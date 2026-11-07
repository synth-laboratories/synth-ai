import json
import stat
import time

import httpx
import pytest

from synth_ai import auth
from synth_ai.errors import SynthError

BASE = "https://api.example.test"
META = {
    "issuer": BASE + "/",
    "code_challenge_methods_supported": ["S256"],
    "token_endpoint": BASE + "/oauth/token",
    "revocation_endpoint": BASE + "/oauth/revoke",
}


@pytest.fixture
def tokens(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(path))
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)
    return path


def test_private_atomic_store_and_environment_precedence(tokens, monkeypatch):
    auth.save_tokens({BASE: {"access_token": "stored", "expires_at": time.time() + 1000}})
    assert stat.S_IMODE(tokens.stat().st_mode) == 0o600
    assert auth.credentials(BASE) == (BASE, "stored")
    monkeypatch.setenv("SYNTH_API_KEY", "environment")
    assert auth.credentials(BASE) == (BASE, "environment")
    assert auth.credentials(BASE, "explicit") == (BASE, "explicit")
    assert "stored" not in json.dumps(auth.status(BASE))
    tokens.chmod(0o644)
    with pytest.raises(SynthError, match="0600"):
        auth.read_tokens()


@pytest.mark.parametrize(
    "metadata",
    [
        dict(META, issuer="https://evil.test"),
        dict(META, code_challenge_methods_supported=["plain"]),
    ],
)
def test_discovery_rejects_mismatched_issuer_and_weak_pkce(metadata):
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=metadata))
    ) as client:
        with pytest.raises(SynthError):
            auth.discover(client, BASE)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.test/token",
        "http://api.example.test/token",
        "https://user@api.example.test/token",
        BASE + "/token#fragment",
    ],
)
def test_endpoint_origin_validation(url):
    with pytest.raises(SynthError):
        auth.endpoint({"token_endpoint": url}, "token_endpoint", BASE)


def test_refresh_rotates_and_logout_revokes_before_removal(tokens, monkeypatch):
    auth.save_tokens(
        {
            BASE: {
                "access_token": "old",
                "refresh_token": "refresh-old",
                "client_id": "dynamic",
                "expires_at": 0,
            }
        }
    )
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=META)
        if request.url.path == "/oauth/token":
            assert b"refresh_token=refresh-old" in request.content
            return httpx.Response(
                200,
                json={"access_token": "new", "refresh_token": "refresh-new", "expires_in": 3600},
            )
        assert b"token=refresh-new" in request.content
        return httpx.Response(200)

    original = httpx.Client
    monkeypatch.setattr(
        auth.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    assert auth.credentials(BASE) == (BASE, "new")
    assert auth.read_tokens()[BASE]["refresh_token"] == "refresh-new"
    auth.logout(BASE)
    assert auth.read_tokens() == {}
    assert len(calls) == 4


def test_production_signs_in_on_the_mcp_host_and_stores_under_the_api_base(tokens, monkeypatch):
    monkeypatch.delenv("SYNTH_OAUTH_BASE_URL", raising=False)
    assert auth.oauth_base(auth.DEFAULT_BASE_URL) == "https://mcp.usesynth.ai"
    assert auth.oauth_base("https://core.internal.test") == "https://core.internal.test"
    monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", "https://other.example.test/")
    assert auth.oauth_base(auth.DEFAULT_BASE_URL) == "https://other.example.test"
    monkeypatch.delenv("SYNTH_OAUTH_BASE_URL")

    oauth = "https://mcp.usesynth.ai"
    seen = {}

    def post(self, url, data=None, **_):
        seen.update(url=url, resource=data["resource"])
        return httpx.Response(
            200,
            json={"access_token": "new", "refresh_token": "r2", "expires_in": 3600},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        auth,
        "discover",
        lambda _client, base: {"token_endpoint": base + "/mcp/oauth/token"},
    )
    monkeypatch.setattr(httpx.Client, "post", post)
    auth.save_tokens(
        {
            auth.DEFAULT_BASE_URL: {
                "access_token": "old",
                "refresh_token": "r1",
                "client_id": "c",
                "expires_at": 0,
            }
        }
    )
    assert auth.credentials(auth.DEFAULT_BASE_URL) == (auth.DEFAULT_BASE_URL, "new")
    assert seen == {"url": oauth + "/mcp/oauth/token", "resource": oauth + "/mcp"}
    assert auth.DEFAULT_BASE_URL in auth.read_tokens()
