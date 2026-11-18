"""Run named journey steps on one driver against a hosted named profile (SYN-5737).

    python -m scripts.acceptance.hosted_journey --journey steps --environment staging \
        --origin https://ORIGIN --driver sdk --steps a,b --receipt-dir NEW_DIR --execute

synth-dev's local_dev.release_tests.hosted_journey invokes this per driver and reads
NEW_DIR/driver-receipt.json. The origin must equal SYNTH_<ENV>_API_BASE_URL, and the
CLI and MCP stdio children select the same named profile. Credentials are the
journey_main names (values never printed). It never launches a run.
Exit 0 all Pass, 1 any Fail, 2 otherwise.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import argparse
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from synth_ai.environments import Environment, EnvironmentProfile

from . import journey_main
from .stripe_checkout import complete_test_checkout
from .surfaces import SurfaceName

HOSTED = (Environment.STAGING.value, Environment.PROD.value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journey", choices=("steps",), required=True)
    parser.add_argument("--environment", choices=HOSTED, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--driver", type=SurfaceName, required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--limit-run-id")
    parser.add_argument("--queued-journal", type=Path)
    parser.add_argument("--foreign-ids", type=Path)
    parser.add_argument("--mcp-url")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args(argv)
    if not options.execute:
        parser.error("--execute is required; planning belongs to the synth-dev caller")
    if options.driver == SurfaceName.MCP_HOSTED and options.mcp_url is None:
        parser.error("mcp_hosted requires --mcp-url")
    variable = "SYNTH_" + options.environment.upper() + "_API_BASE_URL"
    if os.environ.get(variable, "").rstrip("/") != options.origin.rstrip("/"):
        parser.error(variable + " must equal --origin")
    profile = EnvironmentProfile.load(Environment(options.environment))
    if urlsplit(profile.api_base_url).scheme != "https":
        parser.error("hosted journeys require an https origin")
    for name in ("message_fixture", "message_recording_sha256", "message_fixture_sha256"):
        setattr(options, name, None)
    # Stripe test mode exists only on staging; prod checkouts are never completed.
    staging = options.environment == Environment.STAGING.value
    options.complete_checkout = complete_test_checkout if staging else None
    return journey_main.execute(
        options,
        profile.api_base_url,
        profile=options.environment,
        receipt_name="driver-receipt.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
