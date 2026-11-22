"""A harness that owns launch retries passes one-attempt RetryPolicy through the CLI and MCP.

Without it, each CLI or MCP child builds the default three-attempt SDK loop, so one harness
delivery under a retryable overload becomes up to three POSTs (review finding #6).
"""

import json

import httpx
import pytest
from test_cli import api  # noqa: F401 - the CLI fixture environment

from synth_ai import RetryPolicy, cli
from synth_ai.errors import local_problem

ARGV = ("run", "Improve the policy.", "--budget", "10.00", "--idempotency-key", "j1-key", "--json")


def overloaded(requests):
    problem = local_problem("overloaded", "createRun", status=503).model_dump(mode="json")
    problem.update(retryable=True, outcome="not_accepted", service="core")

    def handle(request):
        requests.append(request)
        return httpx.Response(503, headers={"Retry-After": "0"}, json=problem)

    return handle


def launch(capsys, **policy):
    requests = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(overloaded(requests)))
    code = cli.main(list(ARGV), http_client=http, **policy)
    return code, requests, capsys.readouterr().err


@pytest.mark.usefixtures("api")
def test_cli_one_attempt_policy_sends_one_post(capsys):
    code, requests, err = launch(capsys, retry_policy=RetryPolicy(attempts=1, deadline_seconds=5))
    assert code != 0
    assert [r.method for r in requests] == ["POST"]
    assert json.loads(err.strip().splitlines()[-1])["code"] == "overloaded"


@pytest.mark.usefixtures("api")
def test_cli_default_policy_keeps_customer_retries(capsys):
    _, requests, _ = launch(capsys)
    assert len(requests) == 3


def test_mcp_serve_accepts_the_harness_policy():
    import inspect

    from synth_ai.mcp import server

    assert inspect.signature(server.serve).parameters["retry_policy"].default is None
