"""Existing canonical operations must be callable through product SDK/CLI helpers."""

import httpx
import pytest
from test_cli import example

from synth_ai import AsyncSynthClient, cli
from synth_ai._generated.operations import OPERATIONS


@pytest.mark.parametrize(
    "argv",
    [
        ["files", "list"],
        ["files", "get", "file_one"],
        ["uploads", "get", "upl_one"],
        ["keys", "list"],
        ["billing", "plans"],
        ["organization", "usage-summary"],
        ["inspect", "run_one", "--view", "program"],
    ],
)
def test_extended_cli_commands_parse(argv):
    assert cli.parser().parse_args(argv).command == argv[0]


async def test_account_accept_terms_is_a_public_sdk_operation(contract):
    calls = []
    _, body = example(contract, "acceptTerms")

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = await client.account.accept_terms(
                version="2026-11-04", accepted_at="2026-11-16T18:00:00Z"
            )
    assert result.version == "2026-11-04"
    assert calls[0].url.path == OPERATIONS["acceptTerms"]["path"]


RESOURCE_OPERATIONS = [
    "acceptTerms",
    "getUsageSummary",
    "listLimits",
    "createUsageExport",
    "getUsageExport",
    "listApiKeys",
    "createApiKey",
    "updateApiKey",
    "rotateApiKey",
    "revokeApiKey",
    "listFiles",
    "getFile",
    "createUpload",
    "getUpload",
    "finalizeUpload",
    "getRunProgram",
    "getRunConfiguration",
    "listPlans",
    "listBillingHistory",
    "listStatements",
    "getCheckout",
    "createCheckout",
    "getSubscription",
    "cancelSubscription",
    "resumeSubscription",
    "previewSubscriptionChange",
    "changeSubscriptionPlan",
    "createPaymentMethodSession",
    "getAutoReload",
    "setAutoReload",
    "getSpendAlerts",
    "setSpendAlerts",
]
REQUESTS = {
    "acceptTerms": {"version": "2026-11-04", "accepted_at": "2026-11-16T18:00:00Z"},
    "createUsageExport": {
        "since": "2026-11-01T00:00:00Z",
        "until": "2026-11-02T00:00:00Z",
        "format": "csv",
    },
    "createApiKey": {"purpose": "personal"},
    "updateApiKey": {"scopes": ["swarms:read"], "expected_version": 1},
    "createUpload": {
        "name": "task.txt",
        "media_type": "text/plain",
        "size_bytes": 1,
        "digest_sha256": "a" * 64,
    },
    "createCheckout": {"pack_id": "credits.10"},
    "changeSubscriptionPlan": {"plan_id": "max.monthly"},
    "setAutoReload": {"enabled": False},
    "setSpendAlerts": {"enabled": False, "monthly_amount": None},
}


@pytest.mark.parametrize("operation", RESOURCE_OPERATIONS)
async def test_resource_dispatch_uses_existing_wire_contract(contract, operation):
    from synth_ai.mcp.resource_dispatch import dispatch_resource

    op = OPERATIONS[operation]
    route = contract["paths"][op["path"]][op["method"].lower()]
    status = next(code for code in route["responses"] if code.startswith("2"))
    content = route["responses"][status].get("content")
    response_body = None if content is None else content["application/json"]["example"]
    arguments = {name: "fixture_id" for name in op["path_params"]}
    arguments["idempotency_key"] = "resource-once"
    if operation == "previewSubscriptionChange":
        arguments["plan_id"] = "max.monthly"
    if operation in REQUESTS:
        arguments["body"] = REQUESTS[operation]
    calls = []

    def handle(request):
        calls.append(request)
        expected_path = op["path"]
        for name in op["path_params"]:
            expected_path = expected_path.replace("{" + name + "}", arguments[name])
        assert request.url.path == expected_path
        assert request.method == op["method"]
        if operation in REQUESTS:
            import json

            assert json.loads(request.content) == REQUESTS[operation]
        return (
            httpx.Response(int(status), json=response_body)
            if content is not None
            else httpx.Response(int(status))
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = await dispatch_resource(client, operation, arguments)
    assert len(calls) == 1
    assert (
        None if result is None else result.model_dump(mode="json", exclude_unset=True)
    ) == response_body


@pytest.mark.parametrize(
    "argv,operation",
    [
        (["files", "list"], "listFiles"),
        (["uploads", "get", "fixture_id"], "getUpload"),
        (["keys", "list"], "listApiKeys"),
        (["billing", "plans"], "listPlans"),
        (["organization", "usage-summary"], "getUsageSummary"),
        (["inspect", "fixture_id", "--view", "program"], "getRunProgram"),
    ],
)
async def test_cli_resource_reaches_public_sdk(contract, monkeypatch, capsys, argv, operation):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    _, response_body = example(contract, operation)
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.path.startswith(OPERATIONS[operation]["path"].split("{")[0])
        return httpx.Response(200, json=response_body)

    args = cli.parser().parse_args(["--base-url", "https://example.test", *argv, "--json"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        assert await cli.execute(args, http) == 0
    assert len(calls) == 1
    assert capsys.readouterr().out.strip()
