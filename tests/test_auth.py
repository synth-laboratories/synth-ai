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
    assert "stored" not in auth.status(BASE).model_dump_json()
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
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=metadata))
        ) as client,
        pytest.raises(SynthError),
    ):
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
    assert auth.read_tokens()[BASE].refresh_token == "refresh-new"
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
        assert data is not None
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


DEVICE_META = dict(
    META,
    device_authorization_endpoint=BASE + "/oauth/device_authorization",
)


def device_server(monkeypatch, polls, meta=DEVICE_META):
    """Fake OAuth server: device authorization, then the scripted token answers."""
    calls = []
    answers = list(polls)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=meta)
        if request.url.path == "/oauth/device_authorization":
            assert b"client_id=synth-ai-cli" in request.content
            return httpx.Response(
                200,
                json={
                    "device_code": "synth_odc_v1_secret",
                    "user_code": "BCDF-GHJK",
                    "verification_uri": "https://app.example.test/mcp/device",
                    "verification_uri_complete": "https://app.example.test/mcp/device?user_code=BCDF-GHJK",
                    "expires_in": 600,
                    "interval": 5,
                },
            )
        assert b"device_code=synth_odc_v1_secret" in request.content
        assert (
            b"grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code" in request.content
        )
        status, body = answers.pop(0)
        return httpx.Response(status, json=body)

    original = httpx.Client
    monkeypatch.setattr(
        auth.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    return calls


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def test_device_login_polls_through_pending_and_slow_down_then_stores(tokens, monkeypatch):
    monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", BASE)
    device_server(
        monkeypatch,
        [
            (400, {"error": "authorization_pending"}),
            (400, {"error": "slow_down"}),
            (
                200,
                {"access_token": "dev-access", "refresh_token": "dev-refresh", "expires_in": 600},
            ),
        ],
    )
    shown = []
    clock = FakeClock()
    auth.login_device(BASE, show=shown.append, sleep=clock.sleep, clock=clock)
    assert [d.user_code for d in shown] == ["BCDF-GHJK"]
    assert "synth_odc_v1_secret" not in repr(shown[0])
    assert clock.sleeps == [5.0, 5.0, 10.0]
    record = auth.read_tokens()[BASE]
    assert (record.access_token, record.refresh_token, record.client_id) == (
        "dev-access",
        "dev-refresh",
        auth.DEVICE_CLIENT_ID,
    )


@pytest.mark.parametrize(
    ("error", "message"),
    [("access_denied", "denied"), ("expired_token", "expired"), ("invalid_grant", "invalid_grant")],
)
def test_device_login_stops_on_terminal_errors(tokens, monkeypatch, error, message):
    monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", BASE)
    device_server(monkeypatch, [(400, {"error": error})])
    clock = FakeClock()
    with pytest.raises(SynthError, match=message):
        auth.login_device(BASE, show=lambda _d: None, sleep=clock.sleep, clock=clock)
    assert auth.read_tokens() == {}


def test_device_login_gives_up_at_expiry_without_server_help(tokens, monkeypatch):
    monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", BASE)
    device_server(monkeypatch, [(400, {"error": "authorization_pending"})] * 200)
    clock = FakeClock()
    with pytest.raises(SynthError, match="expired"):
        auth.login_device(BASE, show=lambda _d: None, sleep=clock.sleep, clock=clock)
    assert clock.now >= 600


def test_device_login_refuses_a_server_without_device_sign_in(tokens, monkeypatch):
    monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", BASE)
    calls = device_server(monkeypatch, [], meta=META)
    with pytest.raises(SynthError, match="does not offer device sign-in"):
        auth.login_device(BASE, show=lambda _d: None, sleep=lambda _s: None)
    assert [c.method for c in calls] == ["GET"]
