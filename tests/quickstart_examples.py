"""Execute the published quickstart blocks against contract-backed loopback fixtures."""

import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from test_journey import ANSWER, RUN_ID, TASK, FakeApi, route
from test_messages import message, page

from synth_ai.cli import parser

QUICKSTART = Path(__file__).resolve().parents[1] / "docs/quickstart.md"
CLI_ENTRY = "from synth_ai.cli import main; raise SystemExit(main())"
MCP_ENTRY = "from synth_ai.mcp.server import main; main()"


def blocks(language):
    return re.findall(
        rf"^{chr(96) * 3}{language}\n(.*?)^{chr(96) * 3}$", QUICKSTART.read_text(), re.M | re.S
    )


@contextmanager
def fixture_server(contract, directory):
    api = FakeApi(contract)

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            request = httpx.Request(
                self.command,
                f"http://127.0.0.1:{self.server.server_port}{self.path}",
                headers=dict(self.headers),
                content=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
            )
            operation = route(request)
            if operation == "sendRunMessage":
                assert json.loads(request.content)["kind"] == "question"
                response = httpx.Response(202, json=message())
            elif operation == "listRunMessages":
                response = httpx.Response(200, json=page([message(), message(reply=True)]))
            else:
                response = api(request)
            self.send_response(response.status_code)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = do_POST = do_PUT = handle_request

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONNOUSERSITE": "1",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "127.0.0.1",
            "SYNTH_API_KEY": "quickstart-fixture",
            "SYNTH_API_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "SYNTH_AUTH_FILE": str(directory / "absent-auth.json"),
            "SYNTH_STATE_DIR": str(directory / "state"),
        }
        try:
            yield env
        finally:
            server.shutdown()
            worker.join(timeout=5)


def run_example(args, env, directory):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_python_examples(contract, tmp_path):
    examples = blocks("python")
    assert len(examples) == 2
    (tmp_path / "task.md").write_bytes(TASK)
    with fixture_server(contract, tmp_path) as env:
        assert ANSWER.decode().strip() in run_example(["-c", examples[0]], env, tmp_path)
    with fixture_server(contract, tmp_path) as env:
        assert "fixture" in run_example(["-c", examples[1]], env, tmp_path)


def test_cli_example(contract, tmp_path):
    (tmp_path / "task.md").write_bytes(TASK)
    (cli_block,) = blocks("sh")
    statements = cli_block.splitlines()
    assert len(statements) == 7
    run_id = None
    with fixture_server(contract, tmp_path) as env:
        for statement in statements:
            command, *args = shlex.split(statement, comments=True)
            assert command == "synth"
            if "RUN" in args:
                assert run_id == RUN_ID
                args = [run_id if arg == "RUN" else arg for arg in args]
            output = run_example(["-c", CLI_ENTRY, *args], env, tmp_path)
            if args[0] == "run":
                assert RUN_ID in output
                run_id = RUN_ID
            elif args[0] == "result":
                assert ANSWER.decode().strip() in output
    assert (tmp_path / "results/answer.md").read_bytes() == ANSWER


async def test_mcp_example(contract, tmp_path):
    config_text, calls_text = blocks("json")
    config = json.loads(config_text)["mcpServers"]["synth"]
    assert config == {"command": "synth-ai-mcp", "env": {"SYNTH_API_KEY": "..."}}
    calls = json.loads(calls_text)
    assert len(calls) == 7
    replacements = {}
    with fixture_server(contract, tmp_path) as env:
        params = StdioServerParameters(
            command=sys.executable, args=["-c", MCP_ENTRY], env=env, cwd=tmp_path
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                for call in calls:
                    name = call["tool"]
                    args_json = json.dumps(call["arguments"])
                    for key, value in replacements.items():
                        args_json = args_json.replace(json.dumps(key), json.dumps(value))
                    for _ in range(2):
                        result = await session.call_tool(name, json.loads(args_json))
                        assert not result.isError, result.content
                        (content,) = result.content
                        assert isinstance(content, TextContent)
                        value = json.loads(content.text)
                        if name != "swarms_run_wait" or value["outcome"] != "timeout":
                            break
                    else:
                        raise AssertionError("MCP wait never observed completion")
                    if name == "swarms_upload_file":
                        replacements.update(
                            FILE_ID=value["file_id"],
                            REVISION_ID=value["revision_id"],
                            DIGEST_SHA256=value["digest_sha256"],
                        )
                    elif name == "swarms_run_launch":
                        assert value["run_id"] == RUN_ID
                        replacements["RUN"] = value["run_id"]
                    elif name == "swarms_run_result":
                        assert value["answer"] == ANSWER.decode()


def test_current_migration_replacements_parse():
    migration = QUICKSTART.with_name("cli-migration-0.31-0.32.md").read_text()
    current = migration.split("## Moving from 0.31 to 0.41", 1)[1]
    replacements = re.findall(r"^\| `synth [^`]+` \| `synth ([^`]+)` \|$", current, re.M)
    assert len(replacements) == 19
    cli = parser()
    for replacement in replacements:
        cli.parse_args(shlex.split(replacement))
