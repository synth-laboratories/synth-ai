"""One table for the public verbs: each CLI verb has exactly one MCP tool.

Tool names follow the hosted MCP served by synth-backend core (``/mcp``).
``hosted`` records whether the core's hosted catalog served the tool when this
table was last checked against staging; the SDK's local catalog serves all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Verb:
    cli: tuple[str, ...]
    tool: str
    hosted: bool


VERBS: tuple[Verb, ...] = (
    Verb(("swarms", "run"), "swarms_run", False),
    Verb(("swarms", "launch"), "swarms_run_launch", True),
    Verb(("swarms", "status"), "swarms_run_get", True),
    Verb(("swarms", "wait"), "swarms_run_wait", False),
    Verb(("swarms", "events"), "swarms_run_events", True),
    Verb(("swarms", "cancel"), "swarms_run_cancel", True),
    Verb(("swarms", "outputs"), "swarms_run_outputs", True),
    Verb(("swarms", "output"), "swarms_run_output_get", True),
    Verb(("swarms", "list"), "swarms_runs_list", True),
    Verb(("account",), "account_get", True),
    Verb(("billing", "balance"), "account_billing_balance", True),
    Verb(("billing", "history"), "account_billing_history", False),
    Verb(("billing", "plans"), "account_billing_plans", False),
)

TOOL_FOR_CLI = {verb.cli: verb.tool for verb in VERBS}
