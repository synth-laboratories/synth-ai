"""Qualified project-retained Rhodes trace reads on core.v1."""

from .operations import Operation

_PREFIX = "/mnemosyne/projects/{project_id}/native-traces"
LIST = Operation(
    "mnemosyne.native_traces.list",
    "GET",
    _PREFIX,
    True,
    contract="core.v1",
    required_scope="forge:read",
)
GET = Operation(
    "mnemosyne.native_traces.get",
    "GET",
    _PREFIX + "/{identifier}",
    True,
    contract="core.v1",
    required_scope="forge:read",
)
CONTENT = Operation(
    "mnemosyne.native_traces.content",
    "GET",
    _PREFIX + "/{identifier}/content",
    True,
    contract="core.v1",
    required_scope="forge:read",
)
ALL_OPERATIONS = (LIST, GET, CONTENT)
