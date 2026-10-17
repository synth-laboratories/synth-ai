"""Provisional SDK-first Swarms route contract. See SDK surface.md."""

from .operations import Operation

PROJECT = Operation(
    "swarms.projects.get",
    "GET",
    "/swarms/projects/{project_id}",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
PREFLIGHT = Operation(
    "swarms.preflight",
    "POST",
    "/swarms/preflight",
    True,
    contract="swarms.v1",
    required_scope="swarms:run",
)
CREATE = Operation(
    "swarms.create",
    "POST",
    "/swarms/runs",
    True,
    True,
    contract="swarms.v1",
    required_scope="swarms:run",
    may_charge=True,
)
GET = Operation(
    "swarms.get",
    "GET",
    "/swarms/runs/{run_id}",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
LIST = Operation(
    "swarms.list", "GET", "/swarms/runs", True, contract="swarms.v1", required_scope="swarms:read"
)
CONFIGURATION = Operation(
    "swarms.configuration",
    "GET",
    "/swarms/runs/{run_id}/configuration",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
RESULT = Operation(
    "swarms.result",
    "GET",
    "/swarms/runs/{run_id}/result",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
EVENTS = Operation(
    "swarms.events",
    "GET",
    "/swarms/runs/{run_id}/events",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
TASKS = Operation(
    "swarms.tasks",
    "GET",
    "/swarms/runs/{run_id}/tasks",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
CANCEL = Operation(
    "swarms.cancel",
    "POST",
    "/swarms/runs/{run_id}/cancel",
    True,
    True,
    contract="swarms.v1",
    required_scope="swarms:control",
)
CONTROL = Operation(
    "swarms.control",
    "GET",
    "/swarms/runs/{run_id}/controls/{operation_id}",
    True,
    contract="swarms.v1",
    required_scope="swarms:control",
)
USAGE = Operation(
    "swarms.usage",
    "GET",
    "/swarms/runs/{run_id}/usage",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
USAGE_ITEMS = Operation(
    "swarms.usage_operations",
    "GET",
    "/swarms/runs/{run_id}/usage/operations",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
RESOLVE = Operation(
    "swarms.resolve", "GET", "/mutations/resolve", True, contract="swarms.v1", required_scope=None
)
ALL_OPERATIONS = (
    PROJECT,
    PREFLIGHT,
    CREATE,
    GET,
    LIST,
    CONFIGURATION,
    RESULT,
    EVENTS,
    TASKS,
    CANCEL,
    CONTROL,
    USAGE,
    USAGE_ITEMS,
    RESOLVE,
)

OUTPUTS = Operation(
    "swarms.outputs.list",
    "GET",
    "/swarms/runs/{run_id}/outputs",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
OUTPUT = Operation(
    "swarms.outputs.get",
    "GET",
    "/swarms/runs/{run_id}/outputs/{output_id}",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
OUTPUT_CONTENT = Operation(
    "swarms.outputs.content",
    "GET",
    "/swarms/runs/{run_id}/outputs/{output_id}/content",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
ALL_OPERATIONS += (OUTPUTS, OUTPUT, OUTPUT_CONTENT)

FILES = Operation(
    "swarms.files.list",
    "GET",
    "/swarms/projects/{project_id}/files",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
FILE = Operation(
    "swarms.files.get",
    "GET",
    "/swarms/projects/{project_id}/files/{file_id}",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
PREPARE_FILE_UPLOAD = Operation(
    "swarms.files.uploads.prepare",
    "POST",
    "/swarms/projects/{project_id}/uploads",
    True,
    True,
    contract="swarms.v1",
    required_scope="swarms:write",
)
FILE_UPLOAD = Operation(
    "swarms.files.uploads.get",
    "GET",
    "/swarms/uploads/{upload_id}",
    True,
    contract="swarms.v1",
    required_scope="swarms:read",
)
FINALIZE_FILE_UPLOAD = Operation(
    "swarms.files.uploads.finalize",
    "POST",
    "/swarms/uploads/{upload_id}/finalize",
    True,
    True,
    contract="swarms.v1",
    required_scope="swarms:write",
)
ALL_OPERATIONS += (FILES, FILE, PREPARE_FILE_UPLOAD, FILE_UPLOAD, FINALIZE_FILE_UPLOAD)
CREATE_PROJECT = Operation(
    "swarms.projects.create",
    "POST",
    "/swarms/projects",
    True,
    True,
    contract="swarms.v1",
    required_scope="swarms:write",
)
ALL_OPERATIONS += (CREATE_PROJECT,)
