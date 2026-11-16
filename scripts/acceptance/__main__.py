"""Read/download one retained run through SDK or CLI; never launch or spend.

This entrypoint writes raw public-model evidence. It cannot report a complete J1
pass while attempt-tree or result.md evidence is missing. See SYN-5692.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from enum import StrEnum
from pathlib import Path

from synth_ai import Environment, SynthClient

from .j1 import CLIDriver, J1Observation, SDKDriver, observe

__all__ = ["main"]


class Driver(StrEnum):
    SDK = "sdk"
    CLI = "cli"


def persist(observation: J1Observation, directory: Path) -> None:
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
        directory / "observation.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as output:
        output.write(
            json.dumps(
                {
                    "run_id": observation.snapshot.run_id,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Driver, choices=tuple(Driver), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    options = parser.parse_args()
    # These required env contracts are explicit. No keychain, profile fallback,
    # default prod origin, new run or provider request is selected here.
    key = os.environ["SYNTH_API_KEY"]
    origin = os.environ["SYNTH_LOCAL_API_BASE_URL"]
    if not origin.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("read-only local observation requires the leased loopback adapter origin")
    options.receipt_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    if options.driver == Driver.SDK:
        with SynthClient(api_key=key, environment=Environment.LOCAL) as client:
            observation = observe(
                SDKDriver(client),
                run_id=options.run_id,
                output_directory=options.receipt_dir / "outputs",
            )
    else:
        driver = CLIDriver(
            python=Path(sys.executable), environment=os.environ, working_directory=Path.cwd()
        )
        observation = observe(
            driver, run_id=options.run_id, output_directory=options.receipt_dir / "outputs"
        )
    persist(observation, options.receipt_dir)
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
