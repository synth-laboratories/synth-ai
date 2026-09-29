"""`synth-ai index search` route selection: public (anonymous, free) vs keyed (paid).

The CLI mirrors the SDK: the route is chosen explicitly, an inherited
SYNTH_API_KEY never moves `--public` onto the paid route, and a keyed request
without a key is refused instead of being downgraded to public. HTTP is served
by httpx.MockTransport; no network or provider call is made.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from synth_ai.cli import index as cli_index
from synth_ai.cli.main import cli
from synth_ai.client import SynthClient
from synth_ai.sdk.index import PublicIndexClient

BASE = "https://api.example.test"
C1 = "0b6f3c1e-8d2a-4f7b-9c1d-2e3f4a5b6c7d"
SEARCH_ID = "psearch-5f2a"
TOKEN = "tok-do-not-log-8c1d9e"
ENV_KEY = "sk-inherited-env-key"

Handler = Callable[[httpx.Request], httpx.Response]


def _public_capabilities(enabled: bool = True) -> dict[str, Any]:
    limits = {
        "peer_per_minute": 10,
        "peer_per_day": 200,
        "global_per_minute": 600,
        "global_per_day": 20000,
    }
    return {
        "contribution_schema_versions": ["synth.index.contribution.v1"],
        "taxonomy_version": "tax-1",
        "modes": [],
        "search_modes": [],
        "visibilities": ["public"],
        "search_filters": False,
        "private_search": {"activated": False},
        "upload": {"enabled": False},
        "review": {"enabled": False},
        "publication": {"enabled": False},
        "limits": {
            "max_results": 10,
            "max_excerpts_per_result": 2,
            "query_max_bytes": 8192,
            "contents_max_bytes": 65536,
        },
        "public_search": {
            "enabled": enabled,
            "modes": ["fast", "deep"] if enabled else [],
            "limits": {"fast": limits, "deep": limits},
            "daily_budget_cents": 2000,
            "deep_concurrency_max": 2,
            "price_cents": {"fast": 0, "deep": 0},
            "retention": {"public_query_days": 0, "private_processing_minutes": 60},
            "privacy_copy": "Public query and response content is not retained.",
            "token_ttl_seconds": 3600,
            "max_body_bytes": 8192,
        },
    }


def _public_delivery() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": "req-41",
            "mode": "fast",
            "status": "completed",
            "corpus_generation": "corpus-9",
            "ranker_version": "ranker-3",
            "parser_version": "parser-2",
            "taxonomy_version": "tax-1",
            "response": f"Verifiers reward exact program output [{C1}].",
            "citations": [{"contribution_id": C1, "revision_id": "rev-1"}],
            "amount_cents": 0,
        },
        headers={
            "X-Index-Search-Id": SEARCH_ID,
            "X-Search-Token": TOKEN,
            "X-Search-Token-Expires-At": "2026-09-28T13:00:00+00:00",
            "X-Index-Customer-Charge-Cents": "0",
            "X-Index-Monitor-Release": "release-77",
        },
    )


def _paid_result() -> dict[str, Any]:
    versions = {
        "corpus_generation": "corpus-9",
        "ranker_version": "ranker-3",
        "parser_version": "parser-2",
        "taxonomy_version": "tax-1",
    }
    return {
        "search_id": "search-paid-1",
        "request_id": "req-paid-1",
        "requested_mode": "fast",
        "effective_mode": "fast",
        "status": "completed",
        "partial_reason": None,
        **versions,
        "execution_versions": versions,
        "response": f"Verifiers reward exact program output [{C1}].",
        "citations": [{"contribution_id": C1, "revision_id": "rev-1"}],
        "usage": {
            "mode": "fast",
            "billing_scope": "public",
            "price_version": "synth.index.fast.v2",
            "amount_cents": 5,
            "receipt_id": "receipt-1",
            "funding_source": "wallet",
            "wallet_debit_cents": 5,
        },
    }


def _mount(transport_owner: Any, handler: Handler, seen: list[httpx.Request]) -> None:
    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = transport_owner._transport
    old = transport.client
    transport.client = httpx.Client(
        base_url=str(old.base_url), headers=old.headers, transport=httpx.MockTransport(record)
    )


def _backend(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/v1/index/public/capabilities":
        return httpx.Response(200, json=_public_capabilities())
    if path == "/api/v1/index/public/search":
        return _public_delivery()
    if path == "/api/v1/index/search":
        return httpx.Response(200, json=_paid_result())
    return pytest.fail(f"unexpected {request.method} {path}")


class Recorded(list):
    """Requests seen by the mock backend, plus the handler that answers them."""

    handler: Handler = staticmethod(_backend)


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """Route both CLI client constructors to one recorded mock backend."""
    requests = Recorded()

    def public_client(backend_url: str | None) -> PublicIndexClient:
        client = PublicIndexClient(base_url=BASE)
        _mount(client, lambda request: requests.handler(request), requests)
        return client

    def keyed_client(api_key: str, backend_url: str | None) -> SynthClient:
        client = SynthClient(api_key=api_key, base_url=BASE)
        _mount(client.index, lambda request: requests.handler(request), requests)
        return client

    monkeypatch.setattr(cli_index, "open_public_index_client", public_client)
    monkeypatch.setattr(cli_index, "open_keyed_client", keyed_client)
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)
    monkeypatch.delenv("SYNTH_BACKEND_URL", raising=False)
    return requests


def _run(args: list[str], env: dict[str, str] | None = None):
    return CliRunner(mix_stderr=False).invoke(cli, ["index", "search", *args], env=env or {})


# No key -------------------------------------------------------------------------


def test_no_key_and_no_route_is_refused_without_any_request(seen: list[httpx.Request]) -> None:
    result = _run(["RLVR verifier design"])
    assert result.exit_code == 2
    assert "--public" in (result.stdout + result.stderr)
    assert "free anonymous public Search" in (result.stdout + result.stderr)
    assert seen == []


def test_no_key_public_runs_anonymous_free_search_with_terms(seen: list[httpx.Request]) -> None:
    result = _run(["RLVR verifier design", "--public"])
    assert result.exit_code == 0, (result.stdout + result.stderr)
    payload = json.loads(result.stdout)
    assert [request.url.path for request in seen] == [
        "/api/v1/index/public/capabilities",
        "/api/v1/index/public/search",
    ]
    assert all("authorization" not in request.headers for request in seen)
    assert payload["route"] == "public"
    assert payload["customer_charge_cents"] == 0
    assert payload["citations"] == [{"contribution_id": C1, "revision_id": "rev-1"}]
    assert payload["terms"]["available"] is True
    assert payload["terms"]["price"] == "Fast search is free. Deep search is free."
    assert TOKEN not in (result.stdout + result.stderr)


def test_no_key_keyed_is_refused_not_downgraded(seen: list[httpx.Request]) -> None:
    result = _run(["q", "--keyed"])
    assert result.exit_code == 2
    assert "never downgraded to the public route" in (result.stdout + result.stderr)
    assert seen == []


def test_no_key_with_keyed_only_option_is_refused_not_downgraded(
    seen: list[httpx.Request],
) -> None:
    result = _run(["q", "--allow-wallet", "--max-charge-cents", "5"])
    assert result.exit_code == 2
    assert "requires SYNTH_API_KEY" in (result.stdout + result.stderr)
    assert seen == []


# Inherited key ------------------------------------------------------------------


def test_inherited_key_with_public_sends_no_key(seen: list[httpx.Request]) -> None:
    result = _run(["q", "--public"], env={"SYNTH_API_KEY": ENV_KEY})
    assert result.exit_code == 0, (result.stdout + result.stderr)
    assert [request.url.path for request in seen] == [
        "/api/v1/index/public/capabilities",
        "/api/v1/index/public/search",
    ]
    assert all("authorization" not in request.headers for request in seen)
    assert ENV_KEY not in (result.stdout + result.stderr)
    assert "not sent" in result.stderr
    assert json.loads(result.stdout)["route"] == "public"


def test_inherited_key_without_route_uses_keyed_with_a_visible_notice(
    seen: list[httpx.Request],
) -> None:
    result = _run(["q"], env={"SYNTH_API_KEY": ENV_KEY})
    assert result.exit_code == 0, (result.stdout + result.stderr)
    assert [request.url.path for request in seen] == ["/api/v1/index/search"]
    assert seen[0].headers["authorization"] == f"Bearer {ENV_KEY}"
    assert "paid keyed Search because SYNTH_API_KEY is set" in result.stderr
    assert "--public" in result.stderr
    payload = json.loads(result.stdout)
    assert payload["route"] == "keyed"
    assert payload["usage"]["amount_cents"] == 5


def test_inherited_key_with_keyed_is_explicit_and_quiet(seen: list[httpx.Request]) -> None:
    result = _run(
        ["q", "--keyed", "--allow-wallet", "--max-charge-cents", "5"],
        env={"SYNTH_API_KEY": ENV_KEY},
    )
    assert result.exit_code == 0, (result.stdout + result.stderr)
    assert [request.url.path for request in seen] == ["/api/v1/index/search"]
    body = json.loads(seen[0].content)
    assert body["billing"] == {"allow_wallet": True, "max_charge_cents": 5}
    assert result.stderr == ""


# Explicit flags and conflicts -----------------------------------------------------


def test_explicit_api_key_flag_selects_keyed(seen: list[httpx.Request]) -> None:
    result = _run(["q", "--api-key", "sk-flag"])
    assert result.exit_code == 0, (result.stdout + result.stderr)
    assert seen[0].headers["authorization"] == "Bearer sk-flag"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--public", "--keyed"], "mutually exclusive"),
        (["--public", "--api-key", "sk-flag"], "drop --api-key"),
        (["--public", "--allow-wallet"], "--allow-wallet is only for paid keyed Search"),
        (["--public", "--max-charge-cents", "5"], "--max-charge-cents"),
        (["--public", "--private-collection", "col-1"], "--private-collection"),
        (["--public", "--mode", "deep", "--deadline-seconds", "60"], "--deadline-seconds"),
    ],
)
def test_conflicting_route_options_are_refused_before_any_request(
    seen: list[httpx.Request], args: list[str], message: str
) -> None:
    result = _run(["q", *args], env={"SYNTH_API_KEY": ENV_KEY})
    assert result.exit_code == 2, (result.stdout + result.stderr)
    assert message in (result.stdout + result.stderr)
    assert seen == []
    assert ENV_KEY not in (result.stdout + result.stderr)


def test_public_route_disabled_on_backend_fails_closed(seen: Recorded) -> None:
    def disabled(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/index/public/capabilities":
            return httpx.Response(200, json=_public_capabilities(enabled=False))
        return pytest.fail(f"unexpected {request.url.path}")

    seen.handler = disabled
    result = _run(["q", "--public"])
    assert result.exit_code == 1
    assert "not enabled on this backend" in (result.stdout + result.stderr)
    assert [request.url.path for request in seen] == ["/api/v1/index/public/capabilities"]


def test_public_rate_limit_reports_retry_after(seen: Recorded) -> None:
    def limited(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/index/public/capabilities":
            return httpx.Response(200, json=_public_capabilities())
        return httpx.Response(
            429,
            json={"detail": {"code": "index_public_rate_limited", "scope": "peer_minute"}},
            headers={"Retry-After": "12"},
        )

    seen.handler = limited
    result = _run(["q", "--public"])
    assert result.exit_code == 1
    assert "rate limited (scope=peer_minute); retry in 12 s" in (result.stdout + result.stderr)


def test_index_help_does_not_offer_answer() -> None:
    for args in (["index", "--help"], ["index", "search", "--help"]):
        output = CliRunner().invoke(cli, args).output
        assert "answer" not in output.lower(), args
    output = CliRunner().invoke(cli, ["index", "search", "--help"]).output
    assert "--public" in output and "--keyed" in output
