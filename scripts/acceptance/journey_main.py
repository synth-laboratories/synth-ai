"""Run named J2/J3/J-refusal steps on one driver; never launch an uncached run.

    python -m scripts.acceptance.journey_main --capabilities
    python -m scripts.acceptance.journey_main --driver sdk --steps a,b --receipt-dir NEW_DIR

Environment (values never printed): SYNTH_API_KEY, SYNTH_LOCAL_API_BASE_URL and,
per step, SYNTH_JOURNEY_SESSION_TOKEN, SYNTH_JOURNEY_MEMBER_API_KEY,
SYNTH_JOURNEY_EXPIRED_API_KEY, SYNTH_JOURNEY_FOREIGN_API_KEY, and for mcp_hosted
SYNTH_MCP_ACCESS_TOKEN. A missing name makes its steps Inconclusive.
Exit 0 all Pass, 1 any Fail, 2 otherwise. See SYN-5695 and README.md.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import argparse
import json
import os
import sys
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Optional

from synth_ai import SynthClient

from .journeys import JourneyContext, Secret, Step, StepReceipt, Verdict, capabilities, run_step
from .live_limits import retain
from .surfaces import CLISurface, MCPSurface, SDKSurface, Surface, SurfaceName

__all__ = ["main", "surface_factory"]

LOOPBACK = ("http://127.0.0.1:", "http://localhost:")


@contextmanager
def open_surface(
    driver: SurfaceName,
    *,
    origin: str,
    credential: str,
    work_directory: Path,
    mcp_url: Optional[str],
) -> Generator[Surface]:
    environment = {**os.environ, "SYNTH_LOCAL_API_BASE_URL": origin, "SYNTH_API_KEY": credential}
    if driver == SurfaceName.SDK:
        with SynthClient(credential, base_url=origin) as client:
            yield SDKSurface(client)
    elif driver == SurfaceName.CLI:
        yield CLISurface(
            python=Path(sys.executable), environment=environment, work_directory=work_directory
        )
    elif driver == SurfaceName.MCP_STDIO:
        from .mcp_transport import SessionToolCaller, StdioCommand, stdio_session

        command = StdioCommand(
            Path(sys.executable).parent / "synth-ai-mcp",
            ("--env", "local"),
            environment,
            Path.cwd(),
        )
        with SessionToolCaller(stdio_session(command)) as caller:
            yield MCPSurface(caller, name=driver)
    else:
        from .mcp_transport import HostedEndpoint, SessionToolCaller, hosted_session

        if mcp_url is None:
            raise SystemExit("mcp_hosted requires --mcp-url")
        with SessionToolCaller(hosted_session(HostedEndpoint(mcp_url, credential))) as caller:
            yield MCPSurface(caller, name=driver)


def surface_factory(
    driver: SurfaceName, *, origin: str, work_directory: Path, mcp_url: Optional[str]
) -> Callable[[str], AbstractContextManager[Surface]]:
    def make(credential: str) -> AbstractContextManager[Surface]:
        return open_surface(
            driver,
            origin=origin,
            credential=credential,
            work_directory=work_directory,
            mcp_url=mcp_url,
        )

    return make


def summary(receipts: list[StepReceipt]) -> Verdict:
    verdicts = {receipt.verdict for receipt in receipts}
    if Verdict.FAIL in verdicts:
        return Verdict.FAIL
    if verdicts == {Verdict.PASS}:
        return Verdict.PASS
    return Verdict.INCONCLUSIVE


def read_ids(path: Optional[Path]) -> Mapping[str, str]:
    if path is None:
        empty: dict[str, str] = {}
        return empty  # Absent IDs leave the 404 matrix Inconclusive, never Pass.
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise SystemExit("--foreign-ids must be a JSON object of path parameter -> ID")
    ids = {str(key): value for key, value in document.items() if isinstance(value, str)}
    if len(ids) != len(document):
        raise SystemExit("--foreign-ids values must be strings")
    return ids


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--driver", type=SurfaceName, choices=tuple(SurfaceName))
    parser.add_argument("--steps", help="comma-separated step names, executed in order")
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--run-id", help="a running run (raise_limit) or the queued run")
    parser.add_argument("--limit-run-id", help="this driver's run expected to pause on its limit")
    parser.add_argument(
        "--queued-journal",
        type=Path,
        help="cached harness public-client-<kind> directory (launch-row.json)",
    )
    parser.add_argument("--foreign-ids", type=Path, help="org A IDs for the 404 matrix")
    parser.add_argument("--mcp-url")
    options = parser.parse_args(argv)
    if options.capabilities:
        print(json.dumps(capabilities(), sort_keys=True))
        return 0
    if options.driver is None or options.steps is None or options.receipt_dir is None:
        parser.error("--driver, --steps and --receipt-dir are required")
    steps = [Step(name) for name in options.steps.split(",") if name]
    origin = os.environ["SYNTH_LOCAL_API_BASE_URL"].rstrip("/")
    if not origin.startswith(LOOPBACK):
        parser.error("journeys run only against the leased loopback slot origin")
    if options.mcp_url is not None and not options.mcp_url.startswith(LOOPBACK):
        parser.error("--mcp-url must be the leased loopback origin")
    primary = (
        os.environ["SYNTH_MCP_ACCESS_TOKEN"]
        if options.driver == SurfaceName.MCP_HOSTED
        else os.environ["SYNTH_API_KEY"]
    )
    options.receipt_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    secrets = {
        name.value: os.environ[name.value]
        for name in Secret
        if os.environ.get(name.value, "") != ""
    }
    secrets["SYNTH_API_KEY"] = os.environ["SYNTH_API_KEY"]
    make = surface_factory(
        options.driver, origin=origin, work_directory=options.receipt_dir, mcp_url=options.mcp_url
    )
    receipts: list[StepReceipt] = []
    try:
        run_all(options, make, primary, origin, secrets, steps, receipts)
    finally:
        # Written even after an exception, so applied mutations keep their receipts.
        write_steps(options.receipt_dir, options.driver, receipts)
    verdict = summary(receipts)
    print(json.dumps({"driver": options.driver.value, "verdict": verdict.value}))
    return {Verdict.PASS: 0, Verdict.FAIL: 1}.get(verdict, 2)


def write_steps(directory: Path, driver: SurfaceName, receipts: list[StepReceipt]) -> None:
    retain(
        directory / "steps.json",
        json.dumps(
            {
                "schema": "synth.journeys.steps.v1",
                "driver": driver.value,
                "verdict": summary(receipts).value,
                "steps": [receipt.to_json() for receipt in receipts],
            },
            sort_keys=True,
        ),
    )


def run_all(
    options: argparse.Namespace,
    make: Callable[[str], AbstractContextManager[Surface]],
    primary: str,
    origin: str,
    secrets: Mapping[str, str],
    steps: list[Step],
    receipts: list[StepReceipt],
) -> None:
    with make(primary) as surface:
        context = JourneyContext(
            surface=surface,
            driver=options.driver,
            origin=origin,
            make=make,
            receipt_dir=options.receipt_dir,
            run_id=options.run_id,
            limit_run_id=options.limit_run_id,
            queued_journal=options.queued_journal,
            foreign_ids=read_ids(options.foreign_ids),
            secrets=secrets,
            mcp_url=options.mcp_url,
        )
        for step in steps:
            try:
                receipts.append(run_step(context, step))
            except Exception as error:  # A transport or timeout leaves the outcome unknown.
                receipts.append(
                    StepReceipt(
                        step,
                        options.driver,
                        Verdict.INCONCLUSIVE,
                        "step_outcome_unknown:" + type(error).__name__,
                    )
                )


if __name__ == "__main__":
    raise SystemExit(main())
