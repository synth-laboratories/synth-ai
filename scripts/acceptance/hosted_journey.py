"""Hosted binding of existing J1/J2/J3/refusal surface adapters (SYN-5696).

Planning performs no I/O beyond local inputs. Execution requires an explicit
hosted profile/origin and --execute. Local slot entrypoints and their custody
checks remain unchanged. No run is launched by this command.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at command boundaries.
import argparse
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from synth_ai.environments import Environment, EnvironmentProfile

from .__main__ import observe_surface, persist
from .journey_main import run_all, summary, surface_factory, write_steps
from .journeys import Secret, Step
from .parity_view import ParityDriver
from .surfaces import SurfaceName

__all__ = ["main", "validate_target"]


def validate_target(profile: Environment, origin: str, mcp_url: Optional[str]) -> None:
    if profile not in {Environment.STAGING, Environment.PROD}:
        raise ValueError("hosted_profile_required")
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname == ""
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("hosted_exact_https_origin_required")
    if mcp_url is not None:
        mcp = urlsplit(mcp_url)
        if (
            mcp.scheme != "https"
            or mcp.hostname != parsed.hostname
            or mcp.port != parsed.port
            or mcp.username is not None
            or mcp.password is not None
            or mcp.query
            or mcp.fragment
            or mcp.path != "/mcp"
        ):
            raise ValueError("hosted_mcp_origin_mismatch")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        type=Environment,
        choices=(Environment.STAGING, Environment.PROD),
        required=True,
    )
    parser.add_argument("--origin", required=True)
    parser.add_argument("--driver", type=SurfaceName, choices=tuple(SurfaceName), required=True)
    parser.add_argument("--journey", choices=("j1", "steps"), required=True)
    parser.add_argument("--steps", default="")
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--limit-run-id")
    parser.add_argument("--queued-journal", type=Path)
    parser.add_argument("--foreign-ids", type=Path)
    parser.add_argument("--mcp-url")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args(argv)
    validate_target(options.environment, options.origin, options.mcp_url)
    steps = [Step(name) for name in options.steps.split(",") if name]
    if options.journey == "steps" and not steps:
        parser.error("steps journey requires canonical --steps")
    if options.journey == "j1" and not options.run_id:
        parser.error("J1 observes an existing --run-id; never launches")
    if options.receipt_dir.exists():
        parser.error("fresh private receipt directory required")
    if not options.execute:
        print(
            json.dumps(
                {
                    "state": "Planned",
                    "environment": options.environment,
                    "origin": options.origin,
                    "driver": options.driver,
                    "journey": options.journey,
                    "steps": [step.value for step in steps],
                    "network_calls": 0,
                }
            )
        )
        return 0
    variable = (
        "SYNTH_STAGING_API_BASE_URL"
        if options.environment == Environment.STAGING
        else "SYNTH_PROD_API_BASE_URL"
    )
    if variable not in os.environ or EnvironmentProfile.load(
        options.environment
    ).api_base_url != options.origin.rstrip("/"):
        parser.error("explicit named profile origin differs from authorized origin")
    if options.journey == "j1":
        options.receipt_dir.mkdir(mode=0o700)
        observation = observe_surface(
            ParityDriver(options.driver.value),
            key=os.environ["SYNTH_API_KEY"],
            run_id=options.run_id,
            receipt_dir=options.receipt_dir,
            mcp_url=options.mcp_url,
            profile=options.environment,
        )
        persist(ParityDriver(options.driver.value), observation, options.receipt_dir)
        # Same native observer semantics: reads alone never prove containing J1.
        return 2
    options.receipt_dir.mkdir(mode=0o700)
    secrets = {name.value: os.environ[name.value] for name in Secret if name.value in os.environ}
    secrets["SYNTH_API_KEY"] = os.environ["SYNTH_API_KEY"]
    make = surface_factory(
        options.driver,
        origin=options.origin,
        work_directory=options.receipt_dir,
        mcp_url=options.mcp_url,
        profile=options.environment,
    )
    primary = (
        os.environ["SYNTH_MCP_ACCESS_TOKEN"]
        if options.driver == SurfaceName.MCP_HOSTED
        else os.environ["SYNTH_API_KEY"]
    )
    receipts = []
    try:
        run_all(options, make, primary, options.origin, secrets, steps, receipts)
    finally:
        write_steps(options.receipt_dir, options.driver, receipts)
    return {"Pass": 0, "Fail": 1, "Inconclusive": 2}[summary(receipts).value]


if __name__ == "__main__":
    raise SystemExit(main())
