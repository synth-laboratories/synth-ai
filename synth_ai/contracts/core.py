"""SDK-first repository and provenance operations (core.v1)."""

from .operations import Operation

GET_REPOSITORY = Operation(
    "repos.get",
    "GET",
    "/projects/{project_id}/repo",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
REPOSITORY_REFS = Operation(
    "repos.refs",
    "GET",
    "/projects/{project_id}/repo/refs",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
REPOSITORY_COMMIT = Operation(
    "repos.commits.get",
    "GET",
    "/projects/{project_id}/repo/commits/{commit_sha}",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
REPOSITORY_DIFF = Operation(
    "repos.diff",
    "GET",
    "/projects/{project_id}/repo/diff",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
REPOSITORY_ARCHIVE = Operation(
    "repos.archive",
    "GET",
    "/projects/{project_id}/repo/archive",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
CLONE_TOKEN = Operation(
    "repos.clone_tokens.create",
    "POST",
    "/projects/{project_id}/repo/clone-tokens",
    True,
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
LINKS = Operation(
    "links.list", "GET", "/links", True, contract="core.v1", required_scope="swarms:read"
)

ARTIFACT_PREPARE = Operation(
    "artifacts.uploads.prepare",
    "POST",
    "/projects/{project_id}/artifacts/uploads",
    True,
    True,
    contract="core.v1",
    required_scope="swarms:write",
)
ARTIFACT_UPLOAD = Operation(
    "artifacts.uploads.get",
    "GET",
    "/artifacts/uploads/{upload_id}",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
ARTIFACT_FINALIZE = Operation(
    "artifacts.uploads.finalize",
    "POST",
    "/artifacts/uploads/{upload_id}/finalize",
    True,
    True,
    contract="core.v1",
    required_scope="swarms:write",
)
ARTIFACT_GET = Operation(
    "artifacts.get",
    "GET",
    "/projects/{project_id}/artifacts/{artifact_id}",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
ARTIFACT_LIST = Operation(
    "artifacts.list",
    "GET",
    "/projects/{project_id}/artifacts",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
ARTIFACT_CONTENT = Operation(
    "artifacts.content",
    "GET",
    "/projects/{project_id}/artifacts/{artifact_id}/revisions/{revision}/content",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)
ARTIFACT_LABELS = Operation(
    "artifacts.labels.update",
    "PATCH",
    "/projects/{project_id}/artifacts/{artifact_id}/labels",
    True,
    True,
    contract="core.v1",
    required_scope="swarms:write",
)
ARTIFACT_ALIAS = Operation(
    "artifacts.aliases.set",
    "PUT",
    "/projects/{project_id}/artifacts/aliases/{name}",
    True,
    True,
    contract="core.v1",
    required_scope="swarms:write",
)
ARTIFACT_ALIAS_GET = Operation(
    "artifacts.aliases.get",
    "GET",
    "/projects/{project_id}/artifacts/aliases/{name}",
    True,
    contract="core.v1",
    required_scope="swarms:read",
)

ALL_OPERATIONS = (
    GET_REPOSITORY,
    REPOSITORY_REFS,
    REPOSITORY_COMMIT,
    REPOSITORY_DIFF,
    REPOSITORY_ARCHIVE,
    CLONE_TOKEN,
    LINKS,
    ARTIFACT_PREPARE,
    ARTIFACT_UPLOAD,
    ARTIFACT_FINALIZE,
    ARTIFACT_GET,
    ARTIFACT_LIST,
    ARTIFACT_CONTENT,
    ARTIFACT_LABELS,
    ARTIFACT_ALIAS,
    ARTIFACT_ALIAS_GET,
)
