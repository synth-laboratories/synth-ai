"""Read/download one retained run through SDK, CLI or MCP; never launch or spend.

This entrypoint writes raw public-model evidence. It cannot report a complete J1
pass while attempt-tree or result.md evidence is missing. See SYN-5692.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from synth_ai import Environment, SynthClient

from .j1 import CLIDriver, J1Observation, SDKDriver, observe
from .mcp_driver import MCPDriver
from .parity_view import ParityDriver, project

__all__ = ["main"]


LOOPBACK = ("http://127.0.0.1:", "http://localhost:")


def persist(driver: ParityDriver, observation: J1Observation, directory: Path) -> None:
    # An observation gets a fresh, private directory; source artifacts survive
    # retries and cannot be silently overwritten by a later surface read.
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, model in (
        ("snapshot.json", observation.snapshot),
        ("report.json", observation.report),
        ("result.json", observation.result),
    ):
        path = directory / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write(model.model_dump_json() + "\n")
            output.flush()
            os.fsync(output.fileno())
    descriptor = os.open(
        directory / "parity_view.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as output:
        output.write(project(driver, observation).to_json() + "\n")
        output.flush()
        os.fsync(output.fileno())
    descriptor = os.open(
        directory / "observation.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as output:
        output.write(
            json.dumps(
                {
                    "run_id": observation.snapshot.run_id,
                    "driver": driver.value,
                    "result": "Inconclusive",
                    "missing": list(observation.missing),
                    "supplier_charge": "unobserved",
                    "outputs": [
                        {
                            "output_id": item.output.output_id,
                            "path": str(item.path),
                            "sha256": item.sha256,
                        }
                        for item in observation.outputs
                    ],
                },
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        os.fsync(output.fileno())


def observe_surface(
    driver: ParityDriver,
    *,
    key: str,
    run_id: str,
    receipt_dir: Path,
    mcp_url: Optional[str],
    profile: Environment = Environment.LOCAL,
) -> J1Observation:
    match driver:
        case ParityDriver.SDK:
            with SynthClient(api_key=key, environment=profile) as client:
                return observe(
                    SDKDriver(client),
                    run_id=run_id,
                    output_directory=receipt_dir / "outputs",
                )
        case ParityDriver.CLI:
            cli = CLIDriver(
                python=Path(sys.executable),
                environment=os.environ,
                working_directory=Path.cwd(),
                profile=profile,
            )
            return observe(cli, run_id=run_id, output_directory=receipt_dir / "outputs")
        case ParityDriver.MCP_STDIO:
            # Optional extra: synth-ai[mcp] (a streamable_http_client release).
            from .mcp_transport import SessionToolCaller, StdioCommand, stdio_session

            # The stdio server reads origin/credential from its environment only.
            command = StdioCommand(
                Path(sys.executable).parent / "synth-ai-mcp",
                ("--env", profile.value),
                dict(os.environ),
                Path.cwd(),
            )
            with SessionToolCaller(stdio_session(command)) as caller:
                return observe(
                    MCPDriver(caller, surface=driver),
                    run_id=run_id,
                    output_directory=receipt_dir / "outputs",
                )
        case ParityDriver.MCP_HOSTED:
            if mcp_url is None:
                raise SystemExit("mcp_hosted requires --mcp-url")
            from .mcp_transport import HostedEndpoint, SessionToolCaller, hosted_session

            # Opaque bearer for Core /mcp (s.Auth.Authenticate); its credential
            # type is not asserted here. The API key env stays separate.
            endpoint = HostedEndpoint(mcp_url, os.environ["SYNTH_MCP_ACCESS_TOKEN"])
            with SessionToolCaller(hosted_session(endpoint)) as caller:
                return observe(
                    MCPDriver(caller, surface=driver),
                    run_id=run_id,
                    output_directory=receipt_dir / "outputs",
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=ParityDriver, choices=tuple(ParityDriver), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--mcp-url", help="hosted streamable HTTP /mcp endpoint (mcp_hosted)")
    options = parser.parse_args()
    # These required env contracts are explicit. No keychain, profile fallback,
    # default prod origin, new run or provider request is selected here.
    key = os.environ["SYNTH_API_KEY"]
    origin = os.environ["SYNTH_LOCAL_API_BASE_URL"]
    if not origin.startswith(LOOPBACK):
        parser.error("read-only local observation requires the leased loopback adapter origin")
    options.receipt_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    if options.driver == ParityDriver.MCP_HOSTED and (
        options.mcp_url is None or not options.mcp_url.startswith(LOOPBACK)
    ):
        parser.error("mcp_hosted requires --mcp-url on the leased loopback origin")
    observation = observe_surface(
        options.driver,
        key=key,
        run_id=options.run_id,
        receipt_dir=options.receipt_dir,
        mcp_url=options.mcp_url,
    )
    persist(options.driver, observation, options.receipt_dir)
    print(
        json.dumps(
            {
                "run_id": options.run_id,
                "result": "Inconclusive",
                "missing": list(observation.missing),
                "receipt": str(options.receipt_dir / "observation.json"),
            }
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
