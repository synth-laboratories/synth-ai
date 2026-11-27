"""Tests for API version negotiation headers, /v1/version, deprecation warnings, and MCP version."""

from __future__ import annotations

import json
import warnings

import httpx
import pytest
from test_cli import Api, invoke

from synth_ai import (
    ApiVersionInfo,
    ApiVersionUnsupportedError,
    AsyncSynthClient,
    SynthClient,
    __version__,
    cli,
)
from synth_ai._generated.operations import CONTRACT_MAJOR, CONTRACT_VERSION, OPERATIONS
from synth_ai._warnings import reset_warning_cache
from synth_ai.mcp.server import build_server


def _raw_version_example(contract: dict[str, object]) -> dict[str, object]:
    paths = contract["paths"]
    assert isinstance(paths, dict)
    route = paths["/v1/version"]["get"]
    example = route["responses"]["200"]["content"]["application/json"]["example"]
    payload = json.loads(json.dumps(example))
    assert isinstance(payload, dict)
    return {str(k): v for k, v in payload.items()}


def _version_payload(contract: dict[str, object]) -> dict[str, object]:
    payload = _raw_version_example(contract)
    api_obj = payload["api"]
    assert isinstance(api_obj, dict)
    api_obj["version"] = CONTRACT_VERSION
    return payload


async def test_transport_sends_version_and_client_headers(contract: dict[str, object]) -> None:
    captured: list[httpx.Request] = []
    payload = _version_payload(contract)

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            base_url="https://api.example.test",
            http_client=http,
        ) as client:
            info = await client.version()

    assert isinstance(info, ApiVersionInfo)
    assert info.schema_ == "synth.api-discovery.v1"
    assert info.api.version == CONTRACT_VERSION
    assert len(captured) == 1
    req = captured[0]
    assert req.url.path == OPERATIONS["getVersion"]["path"]
    assert "Authorization" not in req.headers
    assert req.headers["Synth-API-Version"] == CONTRACT_VERSION
    assert req.headers["Synth-API-Major"] == str(CONTRACT_MAJOR)
    assert req.headers["X-Synth-Client"].startswith(
        f"synth-ai/{__version__}; contract={CONTRACT_VERSION}; channel="
    )


def test_sync_client_version(contract: dict[str, object]) -> None:
    payload = _version_payload(contract)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/version"
        return httpx.Response(200, json=payload)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with SynthClient(base_url="https://api.example.test", http_client=http) as client:
        info = client.version()
    server = payload["server"]
    assert isinstance(server, dict)
    assert info.server.version == server["version"]


async def test_api_version_unsupported_406_raises_typed_error() -> None:
    problem = {
        "type": "https://docs.usesynth.ai/errors/api_version_unsupported",
        "title": "Not Acceptable",
        "status": 406,
        "detail": "requested API version 1.99.0 is not supported by this server",
        "instance": "urn:request:req-406",
        "code": "api_version_unsupported",
        "service": "synth-backend",
        "stage": "api_version_negotiation",
        "outcome": "not_accepted",
        "retryable": False,
    }

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            406,
            headers={"Content-Type": "application/problem+json", "Synth-API-Version": "1.45.0"},
            json=problem,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture",
            base_url="https://api.example.test",
            http_client=http,
        ) as client:
            with pytest.raises(ApiVersionUnsupportedError) as caught:
                await client.version()
    assert caught.value.problem.code == "api_version_unsupported"
    assert caught.value.problem.status == 406


def test_deprecation_sunset_and_upgrade_warnings_are_deduplicated_and_keep_json_clean(
    contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: object,
) -> None:
    reset_warning_cache()
    payload = _version_payload(contract)
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", f"{tmp_path}/credentials.json")
    monkeypatch.setenv("SYNTH_STATE_DIR", f"{tmp_path}/state")

    api = Api(contract)
    api.overrides["getVersion"] = lambda _req: httpx.Response(
        200,
        headers={
            "Deprecation": "@1795737600",
            "Sunset": "Tue, 01 Jun 2027 00:00:00 GMT",
            "Link": '<https://docs.usesynth.ai/deprecations>; rel="sunset"',
            "X-Synth-Client-Upgrade-Recommended": "0.42.0",
            "X-Synth-Client-Deprecation": "synth-ai 0.41.0 is deprecated; upgrade to 0.42.0",
        },
        json=payload,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        code1, out1, _err1 = invoke(api, capsys, "version", "--server", "--json")
        code2, out2, _err2 = invoke(api, capsys, "version", "--server", "--json")

    assert code1 == cli.EXIT_OK
    assert code2 == cli.EXIT_OK
    parsed1 = json.loads(out1)
    parsed2 = json.loads(out2)
    assert parsed1["schema"] == "synth.api-discovery.v1"
    assert parsed2["schema"] == "synth.api-discovery.v1"
    # Deduplicated: one operation deprecation warning + one client upgrade warning across both calls.
    messages = [str(item.message) for item in caught]
    assert len(messages) == 2
    assert any("getVersion" in m and "Tue, 01 Jun 2027 00:00:00 GMT" in m for m in messages)
    assert any("0.41.0 is deprecated" in m for m in messages)


def test_cli_version_local_and_server(
    contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: object,
) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", f"{tmp_path}/credentials.json")
    monkeypatch.setenv("SYNTH_STATE_DIR", f"{tmp_path}/state")
    api = Api(contract)
    code, out, err = invoke(api, capsys, "version")
    assert code == cli.EXIT_OK, err
    assert __version__ in out and CONTRACT_VERSION in out
    assert api.calls == []

    code, out, err = invoke(api, capsys, "version", "--json")
    assert code == cli.EXIT_OK, err
    local_doc = json.loads(out)
    assert local_doc == {
        "client_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "contract_major": CONTRACT_MAJOR,
    }
    assert api.calls == []

    raw_example = _raw_version_example(contract)
    raw_api = raw_example["api"]
    assert isinstance(raw_api, dict)
    expected_example_version = str(raw_api["version"])
    code, out, err = invoke(api, capsys, "version", "--server")
    assert code == cli.EXIT_OK, err
    assert "server" in out and expected_example_version in out
    assert api.calls[-1][0] == "getVersion"

    code, out, err = invoke(api, capsys, "version", "--server", "--json")
    assert code == cli.EXIT_OK, err
    server_doc = json.loads(out)
    assert server_doc["schema"] == "synth.api-discovery.v1"
    assert server_doc["api"]["version"] == expected_example_version


async def test_stdio_mcp_server_reports_package_version(contract: dict[str, object]) -> None:
    payload = _version_payload(contract)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=payload))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            server = build_server(client, upload_descriptor=None)
            init = server.create_initialization_options()
            assert init.server_version == __version__
