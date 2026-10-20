"""Actual HTTPX routing/CA selection without network or ambient credentials."""

import netrc
import ssl
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.auth import storage
from synth_ai.cli_auth import private_login_file
from synth_ai.core.transfer import async_transfer
from synth_ai.errors import AuthenticationError
from synth_ai.options import ClientOptions, TransferOptions
from tests.http_fixtures import Exchanges, page_response

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def environment(monkeypatch, tmp_path):
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9888")
    path = tmp_path / "netrc"
    path.write_text("machine backend.test login accidental password must-not-send\n")
    monkeypatch.setenv("NETRC", str(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("SDK must not load netrc credentials")

    monkeypatch.setattr(netrc, "netrc", forbidden)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("ca", ["file", "dir", "both"])
async def test_owned_requests_honor_proxy_ca_and_never_netrc(
    monkeypatch, tmp_path, environment, async_mode, bypass, ca
):
    context = ssl.create_default_context()
    certificate = tmp_path / "ca.pem"
    certificate.write_text(ssl.DER_cert_to_PEM_cert(context.get_ca_certs(binary_form=True)[0]))
    directory = tmp_path / "ca-dir"
    directory.mkdir()
    if ca in {"file", "both"}:
        monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    if ca in {"dir", "both"}:
        monkeypatch.setenv("SSL_CERT_DIR", str(directory))
    if bypass:
        monkeypatch.setenv("NO_PROXY", "backend.test")
    create_context = ssl.create_default_context
    contexts = []

    def observed_context(*args, **kwargs):
        contexts.append(kwargs)
        return create_context(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", observed_context)
    handler = Exchanges(page_response())
    routes = []

    async def exchange(transport, request):
        routes.append(type(transport._pool).__name__)
        expected_auth = None if request.url.path.endswith("/capabilities") else "Bearer fixture"
        assert request.headers.get("Authorization") == expected_auth
        response = await httpx.MockTransport(handler).handle_async_request(request)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=httpx.ByteStream(response.content),
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", exchange)
    if async_mode:
        async with AsyncSynthClient(api_key="fixture", base_url="https://backend.test") as client:
            result = await client.index.searches.list(limit=1)
    else:
        with SynthClient(api_key="fixture", base_url="https://backend.test") as client:
            result = client.index.searches.list(limit=1)
    assert result.complete and len(routes) == 2
    assert set(routes) == {"AsyncConnectionPool" if bypass else "AsyncHTTPProxy"}
    expected = {"cafile": str(certificate)} if ca != "dir" else {"capath": str(directory)}
    assert contexts and all(item == expected for item in contexts)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_borrowed_auth_does_not_replace_sdk_key(async_mode):
    handler = Exchanges(page_response())
    if async_mode:
        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(handler), auth=("wrong", "wrong")
            ) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            await client.index.searches.list(limit=1)
    else:
        with (
            httpx.Client(
                transport=httpx.MockTransport(handler), auth=("wrong", "wrong")
            ) as network,
            SynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(http_client=network),
            ) as client,
        ):
            client.index.searches.list(limit=1)
    assert all("Authorization" not in request.headers for request in handler.capability_requests)
    assert all(request.headers["Authorization"] == "Bearer fixture" for request in handler.requests)


async def test_signed_upload_honors_proxy_without_api_or_netrc_credentials(
    monkeypatch, tmp_path, environment
):
    path = tmp_path / "upload.txt"
    data = b"public upload fixture"
    path.write_bytes(data)
    manifest = SimpleNamespace(
        byte_size=len(data), digest_sha256=sha256(data).hexdigest(), media_type="text/plain"
    )
    source = SimpleNamespace(path=path, manifest=manifest)
    status = SimpleNamespace(
        state="prepared",
        upload_id="upload",
        manifest=manifest,
        transfer_url="https://backend.test/upload",
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
        transfer_headers={},
    )
    requests = []

    async def exchange(transport, request):
        assert type(transport._pool).__name__ == "AsyncHTTPProxy"
        assert "Authorization" not in request.headers
        assert await request.aread() == data
        requests.append(request)
        return await httpx.MockTransport(lambda _: httpx.Response(200)).handle_async_request(
            request
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", exchange)
    receipt = await async_transfer(
        status,
        source,
        TransferOptions(timeout_seconds=5, maximum_bytes=1024, chunk_bytes=1024, retry_count_max=0),
        ("backend.test",),
    )
    assert receipt.digest_sha256 == manifest.digest_sha256 and len(requests) == 1


@pytest.mark.parametrize("entrypoint", ["store", "login"])
async def test_private_files_refuse_non_posix_before_getuid_or_file_io(
    monkeypatch, tmp_path, entrypoint
):
    path = tmp_path / "credentials.json"
    # Replace only this module's os binding; pathlib/pytest keep the host platform.
    monkeypatch.setattr(storage, "os", SimpleNamespace(name="nt"))
    with pytest.raises(AuthenticationError) as caught:
        if entrypoint == "store":
            storage.FileCredentialStore(path)
        else:
            private_login_file("fixture one-time URL")
    assert caught.value.code == "credential_storage_unsupported"
    assert caught.value.stage == "credential_storage" and "api_key" in str(caught.value)
    assert not path.exists()
