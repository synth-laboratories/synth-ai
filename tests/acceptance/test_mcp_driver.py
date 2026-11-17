"""MCP J1 driver: fake-transport T0 plus one real stdio child on fixture HTTP.

No provider, slot, staging or prod is used. The fake caller returns exactly the
text the stdio server emits (public model JSON) or Core's hosted failure shape.
See SYN-5693.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_j1_drivers import RUN, fixture_api

pytest.importorskip("mcp")

from scripts.acceptance.j1 import AdapterError, Missing, SDKDriver, observe, observe_listed
from scripts.acceptance.mcp_driver import MCPDriver, MCPRefusal, Tool, ToolOutcome
from scripts.acceptance.mcp_transport import SessionToolCaller, StdioCommand, stdio_session
from scripts.acceptance.parity_view import ParityDriver, project
from synth_ai import SynthClient

pytestmark = pytest.mark.acceptance
ROOT = Path(__file__).parents[2]
PIN = "b" * 64


def coherent_examples(contract):
    examples = {}
    for path in ("/v1/runs/{run_id}", "/v1/runs/{run_id}/report", "/v1/runs/{run_id}/result"):
        examples[path] = contract["paths"][path]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    examples = json.loads(json.dumps(examples))
    examples["/v1/runs/{run_id}/report"]["run_id"] = RUN
    examples["/v1/runs/{run_id}/result"]["charge"]["amount"] = examples["/v1/runs/{run_id}/report"][
        "settlement"
    ]["total"]
    return examples


def check_arguments(name, arguments):
    # Same contract the real servers enforce: the catalog inputSchema.
    from jsonschema import Draft202012Validator

    from synth_ai.mcp.tools import TOOLS

    [tool] = [tool for tool in TOOLS if tool["name"] == name]
    Draft202012Validator(tool["inputSchema"]).validate(arguments)


class FakeCaller:
    """Records schema-valid calls; answers with stdio-server-shaped text per tool."""

    def __init__(self, answers, *, names=None):
        self.answers = answers
        self.names = frozenset(answers) if names is None else names
        self.calls = []

    def tools(self):
        return self.names

    def call(self, name, arguments):
        check_arguments(name, dict(arguments))
        self.calls.append((name, dict(arguments)))
        answer = self.answers[name]
        if isinstance(answer, list):
            answer = answer.pop(0)
        return answer if isinstance(answer, ToolOutcome) else ToolOutcome(False, json.dumps(answer))


def mcp_answers(contract, *, outputs=None):
    examples = coherent_examples(contract)
    return {
        Tool.GET: examples["/v1/runs/{run_id}"],
        Tool.REPORT: examples["/v1/runs/{run_id}/report"],
        Tool.RESULT: examples["/v1/runs/{run_id}/result"],
        Tool.OUTPUTS: {"has_more": False, "items": [] if outputs is None else outputs},
    }


def test_fake_mcp_maps_tool_results_into_same_observation_as_sdk(contract, tmp_path):
    with (
        fixture_api(contract, coherent=True) as (origin, _),
        SynthClient("fixture", base_url=origin) as client,
    ):
        sdk = observe(SDKDriver(client), run_id=RUN, output_directory=tmp_path / "sdk")
    caller = FakeCaller(mcp_answers(contract))
    mcp = observe_listed(MCPDriver(caller, surface="mcp_stdio"), run_id=RUN)
    assert (mcp.snapshot, mcp.report, mcp.result, mcp.listed) == (
        sdk.snapshot,
        sdk.report,
        sdk.result,
        sdk.listed,
    )
    assert set(sdk.missing) <= set(mcp.missing)
    sdk_view = project(ParityDriver.SDK, sdk)
    mcp_view = project(ParityDriver.MCP_STDIO, mcp)
    assert sdk_view.charge_usd is not None
    assert json.loads(sdk_view.to_json()) | {"driver": "x"} == json.loads(mcp_view.to_json()) | {
        "driver": "x"
    }
    assert [name for name, _ in caller.calls] == [Tool.GET, Tool.REPORT, Tool.RESULT, Tool.OUTPUTS]
    assert caller.calls[0][1] == {"run_id": RUN}
    assert caller.calls[3][1] == {"run_id": RUN, "limit": 100}


def test_listed_outputs_without_content_tool_are_named_not_fabricated(contract):
    output = {
        "run_id": RUN,
        "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
        "state": "ready",
        "reference": {"revision_id": "rev_fixture", "digest_sha256": PIN},
        "filename": "result.md",
    }
    caller = FakeCaller(mcp_answers(contract, outputs=[output]))
    observation = observe_listed(MCPDriver(caller, surface="mcp_hosted"), run_id=RUN)
    assert [item.output_id for item in observation.listed] == ["out_AAAAAAAAAAAAAAAAAAAAAA"]
    assert observation.outputs == ()
    assert Missing.OUTPUT_CONTENT in observation.missing
    assert Missing.RESULT_MARKDOWN in observation.missing


def test_tool_absent_from_server_list_is_typed_refusal_without_call(contract):
    answers = mcp_answers(contract)
    caller = FakeCaller(answers, names=frozenset(answers) - {Tool.RESULT})
    with pytest.raises(AdapterError) as failure:
        observe_listed(MCPDriver(caller, surface="mcp_stdio"), run_id=RUN)
    assert failure.value.code == MCPRefusal.TOOL_UNAVAILABLE
    assert failure.value.stage == Tool.RESULT
    assert Tool.RESULT not in [name for name, _ in caller.calls]


PROBLEM = {
    "type": "https://docs.usesynth.ai/problems/run_not_found",
    "title": "Run not found",
    "status": 404,
    "detail": "secret-bearing detail must not be copied",
    "instance": "/v1/runs/" + RUN,
    "code": "run_not_found",
    "service": "core",
    "stage": "runs",
    "outcome": "not_accepted",
}


@pytest.mark.parametrize(
    ("text", "code", "cause"),
    [
        # stdio server: SwarmsAPIError -> isError with the Problem JSON.
        (json.dumps(PROBLEM), MCPRefusal.TOOL_REFUSED, "problem=run_not_found status=404"),
        # Core hosted /mcp: Problem nested under data.response.
        (
            json.dumps(
                {
                    "error": "Operation refused.",
                    "data": {"tool": "swarms_run_get", "status": 404, "response": PROBLEM},
                },
                indent=2,
            ),
            MCPRefusal.TOOL_REFUSED,
            "problem=run_not_found status=404",
        ),
        # Core hosted /mcp argument refusal: fixed message, no Problem.
        (
            json.dumps(
                {
                    "error": "Unknown tool argument.",
                    "data": {"tool": "swarms_run_get", "argument": "x"},
                }
            ),
            MCPRefusal.TOOL_FAILED,
            "error=Unknown tool argument. status=none",
        ),
        # stdio server ValueError text (e.g. "Unknown tool").
        ("Unknown tool", MCPRefusal.TOOL_FAILED, "unstructured tool error"),
    ],
)
def test_failed_tool_calls_surface_as_typed_refusals(contract, text, code, cause):
    answers = mcp_answers(contract)
    answers[Tool.GET] = ToolOutcome(True, text)
    with pytest.raises(AdapterError) as failure:
        MCPDriver(FakeCaller(answers), surface="mcp_stdio").snapshot(RUN)
    assert (failure.value.code, failure.value.stage, failure.value.cause) == (code, Tool.GET, cause)
    assert "secret-bearing" not in failure.value.cause


def test_success_with_wrong_model_is_typed_refusal(contract):
    answers = mcp_answers(contract)
    answers[Tool.REPORT] = answers[Tool.GET]
    with pytest.raises(AdapterError) as failure:
        MCPDriver(FakeCaller(answers), surface="mcp_stdio").report(RUN)
    assert failure.value.code == MCPRefusal.RESULT_INVALID


def test_wait_loops_bounded_server_waits_until_settled(contract):
    wait = contract["paths"]["/v1/runs/{run_id}/wait"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    settled = json.loads(json.dumps(wait)) | {"outcome": "terminal"}
    settled["run"]["state"] = "completed"
    caller = FakeCaller({Tool.WAIT: [wait, wait, settled]})
    snapshot = MCPDriver(caller, surface="mcp_stdio").wait(RUN)
    assert snapshot.state == "completed"
    assert [arguments["timeout_seconds"] for _, arguments in caller.calls] == [50, 50, 50]


def test_wait_that_never_settles_is_typed_refusal(contract):
    wait = contract["paths"]["/v1/runs/{run_id}/wait"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    caller = FakeCaller({Tool.WAIT: [wait] * 11})
    with pytest.raises(AdapterError) as failure:
        MCPDriver(caller, surface="mcp_stdio").wait(RUN)
    assert failure.value.code == MCPRefusal.WAIT_UNSETTLED
    assert len(caller.calls) == 11


def test_real_stdio_child_reads_same_models_as_sdk(contract, tmp_path):
    executable = Path(sys.executable).parent / "synth-ai-mcp"
    with fixture_api(contract, coherent=True) as (origin, requests):
        with SynthClient("fixture", base_url=origin) as client:
            sdk = observe(SDKDriver(client), run_id=RUN, output_directory=tmp_path / "sdk")
        command = StdioCommand(
            executable,
            ("--env", "local"),
            {"SYNTH_LOCAL_API_BASE_URL": origin, "SYNTH_API_KEY": "fixture", "HOME": str(tmp_path)},
            ROOT,
        )
        with SessionToolCaller(stdio_session(command)) as caller:
            assert {str(tool) for tool in Tool} <= caller.tools()
            mcp = observe_listed(MCPDriver(caller, surface="mcp_stdio"), run_id=RUN)
            unknown = caller.call("swarms_run_teleport", {"run_id": RUN})
    assert (mcp.snapshot, mcp.report, mcp.result, mcp.listed) == (
        sdk.snapshot,
        sdk.report,
        sdk.result,
        sdk.listed,
    )
    assert (
        project(ParityDriver.MCP_STDIO, mcp).charge_usd == project(ParityDriver.SDK, sdk).charge_usd
    )
    assert unknown.is_error
    with pytest.raises(AdapterError) as failure:
        from scripts.acceptance.mcp_driver import classify_failure

        raise classify_failure(Tool.GET, unknown, run_id=RUN)
    assert failure.value.code == MCPRefusal.TOOL_FAILED
    # SDK 4 reads + MCP child 4 reads, all against the fixture origin.
    assert len(requests) == 8


def test_launch_auto_is_one_journaled_tool_call(contract, tmp_path):
    from scripts.acceptance.launch_journal import LaunchJournal

    snapshot = coherent_examples(contract)["/v1/runs/{run_id}"] | {
        "optimizer": {
            "topology": "swarms_v2",
            "version": 2,
            "params": {},
            "reason": "environment_default",
            "definition_digest": "sha256:" + "a" * 64,
        }
    }
    caller = FakeCaller({Tool.LAUNCH: snapshot})
    accepted = MCPDriver(caller, surface="mcp_stdio").launch_auto(
        message="task",
        budget_usd="1.00",
        files=(),
        idempotency_key="j1-mcp-once",
        journal=LaunchJournal(tmp_path / "launch"),
    )
    assert accepted.run_id == RUN
    [(name, arguments)] = caller.calls
    assert name == Tool.LAUNCH
    assert arguments["idempotency_key"] == "j1-mcp-once"
    assert arguments["body"]["optimizer"] == "auto"
    assert arguments["body"]["message"] == "task"


def test_upload_refuses_formats_the_catalog_cannot_carry(tmp_path):
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"\0")
    caller = FakeCaller({Tool.UPLOAD: {}})
    with pytest.raises(AdapterError) as failure:
        MCPDriver(caller, surface="mcp_stdio").upload(binary, idempotency_key="u1")
    assert failure.value.code == MCPRefusal.UPLOAD_MEDIA_TYPE
    assert caller.calls == []


def test_usage_for_run_reads_one_bounded_page():
    caller = FakeCaller({Tool.USAGE: {"has_more": False, "items": []}})
    page = MCPDriver(caller, surface="mcp_stdio").usage(RUN)
    assert page.items == []
    assert caller.calls == [(Tool.USAGE, {"run_id": RUN, "limit": 100})]


@pytest.mark.parametrize("body", [b"# result\n", b"\x00\xff\x80", b""])
def test_output_content_preserves_exact_bytes(body, tmp_path):
    from base64 import b64encode
    from hashlib import sha256

    from synth_ai.models import RunOutput

    digest = sha256(body).hexdigest()
    output = RunOutput(
        run_id=RUN,
        output_id="out_AAAAAAAAAAAAAAAAAAAAAA",
        state="ready",
        filename="result.md",
        reference={"revision_id": "rev_fixture", "digest_sha256": digest},
    )
    caller = FakeCaller(
        {
            Tool.CONTENT: {
                "encoding": "base64",
                "content": b64encode(body).decode(),
                "media_type": "application/octet-stream",
                "byte_size": len(body),
                "sha256": digest,
            }
        }
    )
    target = tmp_path / "result.md"
    assert MCPDriver(caller, surface="mcp_stdio").download(RUN, output, target).read_bytes() == body
    with pytest.raises(FileExistsError):
        MCPDriver(caller, surface="mcp_stdio").download(RUN, output, target)


@pytest.mark.parametrize(
    "change",
    [{"content": "!!!!"}, {"content": "Yg=="}, {"byte_size": 2}, {"sha256": "b" * 64}],
)
def test_output_mismatch_refuses_before_writing(change, tmp_path):
    from hashlib import sha256

    from synth_ai.models import RunOutput

    digest = sha256(b"a").hexdigest()
    output = RunOutput(
        run_id=RUN,
        output_id="out_AAAAAAAAAAAAAAAAAAAAAA",
        state="ready",
        filename="result.md",
        reference={"revision_id": "rev_fixture", "digest_sha256": digest},
    )
    envelope = {
        "encoding": "base64",
        "content": "YQ==",
        "media_type": "application/octet-stream",
        "byte_size": 1,
        "sha256": digest,
    } | change
    caller = FakeCaller({Tool.CONTENT: envelope})
    with pytest.raises(AdapterError) as failure:
        MCPDriver(caller, surface="mcp_hosted").download(RUN, output, tmp_path / "result.md")
    assert failure.value.code == MCPRefusal.RESULT_INVALID
    assert not (tmp_path / "result.md").exists()


def test_adapter_refusal_survives_generator_context_manager():
    from contextlib import contextmanager

    @contextmanager
    def scope():
        yield

    error = AdapterError("fixture", "wait", "refused", RUN)
    with pytest.raises(AdapterError) as failure, scope():
        raise error
    assert failure.value is error


@pytest.mark.parametrize("surface", ["mcp_stdio", "mcp_hosted"])
def test_mcp_recorded_launch_preserves_public_body_and_refuses_repeat(contract, tmp_path, surface):
    from test_j1_drivers import RECORDED_BODY, RECORDED_PUBLIC

    from scripts.acceptance.j1 import RecordedLaunch
    from scripts.acceptance.launch_journal import LaunchJournal, LaunchJournalError

    answers = coherent_examples(contract)
    caller = FakeCaller({Tool.LAUNCH: answers["/v1/runs/{run_id}"]})
    driver = MCPDriver(caller, surface=surface)
    journal = LaunchJournal(tmp_path / "launch")
    recording = RecordedLaunch.from_body(RECORDED_BODY)
    accepted = driver.launch_recorded(
        recording=recording, idempotency_key="recorded-once", journal=journal
    )
    assert journal.read_accepted() == accepted
    assert caller.calls == [
        (Tool.LAUNCH, {"body": RECORDED_PUBLIC, "idempotency_key": "recorded-once"})
    ]
    with pytest.raises(LaunchJournalError):
        driver.launch_recorded(
            recording=recording, idempotency_key="recorded-once", journal=journal
        )
    assert len(caller.calls) == 1


def test_mcp_recorded_launch_keeps_intent_after_transport_failure(tmp_path):
    from test_j1_drivers import RECORDED_BODY

    from scripts.acceptance.j1 import RecordedLaunch
    from scripts.acceptance.launch_journal import LaunchJournal, LaunchJournalError

    class LostReply(FakeCaller):
        def call(self, name, arguments):
            raise AdapterError("mcp_transport_failed", str(name), "reply lost", RUN)

    caller = LostReply({}, names=frozenset({Tool.LAUNCH}))
    driver = MCPDriver(caller, surface="mcp_hosted")
    journal = LaunchJournal(tmp_path / "launch")
    recording = RecordedLaunch.from_body(RECORDED_BODY)
    with pytest.raises(AdapterError):
        driver.launch_recorded(recording=recording, idempotency_key="uncertain", journal=journal)
    assert (journal.directory / "launch-intent.json").is_file()
    assert not (journal.directory / "launch-accepted.json").exists()
    with pytest.raises(LaunchJournalError):
        driver.launch_recorded(recording=recording, idempotency_key="uncertain", journal=journal)


def test_mcp_utf8_output_preserves_unicode_bytes(tmp_path):
    from hashlib import sha256

    from synth_ai.models import RunOutput

    body = "Résumé 🦀\n".encode()
    digest = sha256(body).hexdigest()
    output = RunOutput(
        run_id=RUN,
        output_id="out_AAAAAAAAAAAAAAAAAAAAAA",
        state="ready",
        filename="result.md",
        reference={"revision_id": "rev_fixture", "digest_sha256": digest},
    )
    caller = FakeCaller(
        {
            Tool.CONTENT: {
                "encoding": "utf8",
                "content": body.decode(),
                "media_type": "text/markdown",
                "byte_size": len(body),
                "sha256": digest,
            }
        }
    )
    assert (
        MCPDriver(caller, surface="mcp_hosted")
        .download(RUN, output, tmp_path / "result.md")
        .read_bytes()
        == body
    )
