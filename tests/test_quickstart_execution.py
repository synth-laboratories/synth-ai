"""Run the actual published snippets through the installed SDK and CLI."""

import json
import re
import shlex
from pathlib import Path

import httpx
from test_journey import RUN_ID, TASK, FakeApi, route
from test_messages import message, page

from synth_ai import cli
from synth_ai.mcp.tools import TOOLS

DOC = Path(__file__).resolve().parents[1] / "docs/quickstart.md"


def test_published_quickstart(contract, tmp_path, monkeypatch):
    markdown = DOC.read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTH_API_KEY", "offline-fixture")
    monkeypatch.setenv("SYNTH_BASE_URL", "https://example.test")
    (tmp_path / "task.md").write_bytes(TASK)
    owner = FakeApi(contract)

    def respond(req):
        if req.method != "PUT":
            op = route(req)
            if op == "sendRunMessage":
                return httpx.Response(202, json=message())
            if op == "listRunMessages":
                return httpx.Response(200, json=page([message(), message(reply=True)]))
        return owner(req)

    # Inject the transport at the HTTP constructor. All product entry points,
    # including the sync bridge, still execute their normal request code.
    constructor = httpx.AsyncClient.__init__

    def offline(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        return constructor(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", offline)

    examples = re.findall(r"```python\n(.*?)```", markdown, re.S)
    assert len(examples) == 2
    for source in examples:
        exec(compile(source, str(DOC), "exec"), {})
    (block,) = re.findall(r"```sh\n(.*?)```", markdown, re.S)
    for line in block.strip().splitlines():
        exe, *args = shlex.split(line, comments=True)
        assert exe == "synth"
        assert cli.main([RUN_ID if word == "RUN" else word for word in args]) == cli.EXIT_OK, line

    (conf,) = re.findall(r"```json\n(.*?)```", markdown, re.S)
    assert json.loads(conf)["mcpServers"]["synth"]["command"] == "synth-ai-mcp"
    tools = re.findall(r"`((?:swarms_|account_usage_)[a-z_]+)`", markdown.split("## MCP")[1])
    assert tools and set(tools) <= {tool["name"] for tool in TOOLS}


def test_documented_migration_cli_destinations_parse():
    guide = (DOC.parent / "cli-migration-0.31-0.32.md").read_text()
    targets = [
        command
        for line in guide.splitlines()
        if line.startswith("| `synth ")
        for command in re.findall(r"`synth ([^`]+)`", line.split("|")[2])
    ]
    assert targets
    for command in targets:
        cli.parser().parse_args(["1" if arg == "USD" else arg for arg in shlex.split(command)])
