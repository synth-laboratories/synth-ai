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

ALL_OPERATIONS = (
    GET_REPOSITORY,
    REPOSITORY_REFS,
    REPOSITORY_COMMIT,
    REPOSITORY_DIFF,
    REPOSITORY_ARCHIVE,
    CLONE_TOKEN,
    LINKS,
)
