"""Project one surface's J1 observation into the same-run parity record.

See SYN-5693. synth-dev `local_dev/release_tests/parity.py` loads this exact
JSON (schema `synth.j1.parity_view.v1`) as one `Observation` per driver. The
attempt tree stays null while no public attempt-tree contract exists; null is
"unavailable", never an empty tree. Charge is the customer charge already
cross-checked between result and report by `observe_records`.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from .j1 import J1Observation

__all__ = ["SCHEMA", "ParityDriver", "ParityView", "project"]
SCHEMA = "synth.j1.parity_view.v1"


class ParityDriver(StrEnum):
    # Values equal synth-dev local_dev.release_tests.journeys.Driver.
    SDK = "sdk"
    CLI = "cli"
    MCP_STDIO = "mcp_stdio"
    MCP_HOSTED = "mcp_hosted"


@dataclass(frozen=True, slots=True)
class ParityView:
    driver: ParityDriver
    run_id: str
    state: str
    charge_usd: Optional[str]
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": SCHEMA,
                "driver": self.driver.value,
                "run_id": self.run_id,
                "state": self.state,
                "charge_usd": self.charge_usd,
                "attempts": None,
                "execution_closed": self.execution_closed,
                "accounting_final": self.accounting_final,
                "cleanup_complete": self.cleanup_complete,
            },
            sort_keys=True,
        )


def project(driver: ParityDriver, observation: J1Observation) -> ParityView:
    charge = observation.result.charge
    snapshot = observation.snapshot
    return ParityView(
        driver,
        snapshot.run_id,
        snapshot.state,
        None if charge is None else charge.amount.amount,
        snapshot.execution_closed,
        snapshot.accounting_final,
        snapshot.cleanup_complete,
    )
