"""Candidate project evidence operations; backend capability admission is required."""

from .operations import Operation

_PREFIX = "/mnemosyne/projects/{project_id}/evaluation-evidence"
CAPABILITIES = Operation(
    "mnemosyne.evidence.capabilities",
    "GET",
    _PREFIX + "-capabilities",
    True,
    contract="core.v1",
    required_scope="forge:read",
)
LIST = Operation(
    "mnemosyne.evidence.list", "GET", _PREFIX, True, contract="core.v1", required_scope="forge:read"
)
GET = Operation(
    "mnemosyne.evidence.get",
    "GET",
    _PREFIX + "/{identifier}",
    True,
    contract="core.v1",
    required_scope="forge:read",
)
MEASUREMENTS = Operation(
    "mnemosyne.evidence.measurements",
    "GET",
    _PREFIX + "/{identifier}/measurements",
    True,
    contract="core.v1",
    required_scope="forge:read",
)
COMPARE = Operation(
    "mnemosyne.evidence.compare",
    "GET",
    _PREFIX + "-comparison",
    True,
    contract="core.v1",
    required_scope="forge:read",
)
ALL_OPERATIONS = (CAPABILITIES, LIST, GET, MEASUREMENTS, COMPARE)
