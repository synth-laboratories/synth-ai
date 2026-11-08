# Typing and data protocol standards

Core's pinned OpenAPI remains the authority. Update it before changing a wire
shape; regenerate models, operations, MCP schemas, and synchronous delegates.
There is no second handwritten set of request models.

## Python records and boundaries

Public methods have explicit typed parameters. Use `RunInputs`/`InputFile`,
`OwnerResourceRequest`/`PoolSelection`, `RoutingConstraints`, `RunGuidance`, and
`Money` instead of dictionaries describing those domain objects. The generated
schema enums are `StrEnum` classes: values serialize as their existing strings.
For example, `RunGuidanceKind.QUESTION` and `RunGuidancePriority.MUST`.
Raw JSON remains accepted by `LaunchRequest.model_validate` and the CLI JSON
boundary, where it undergoes the same validation.

Generated records are frozen: field reassignment is refused. Their nested list
and map values are ordinary Python collections, so freezing is shallow. Treat
records as values; construct and validate a new record to change one. Pydantic's
`model_construct` and `model_copy(update=...)` bypass validation and are not input
validation APIs.

Validation is strict: booleans, numeric strings and fractional numbers cannot
stand in for integer limits. Invalid input fails before sending a request. A
malformed successful response raises `ResponseContractError` and retains request
identity and mutation uncertainty. Missing optional values stay distinguishable
from wire fields explicitly set to null.

Arbitrary JSON is limited to fields the upstream contract intentionally leaves
open, such as metadata and extension payloads. Those use recursive `JsonValue`,
not `Any`, and reject non-JSON objects. Response extensions stay forward compatible.
Closed schemas reject unknown properties.

## Lifecycle and semantic events

The patterns are informed by [OpenResponses](https://www.openresponses.org/specification):
explicit records, finite known states, semantic events, and structured errors.
Synth remains its own API; this does not claim OpenResponses wire compatibility.

`CompletionMilestone.FINISHED` requires `execution_closed`, `accounting_final`,
and `cleanup_complete`. `RunSnapshot.finished` and `.reached(milestone)` share
that definition with polling. `.known_state` returns a `RunState` enum or `None`;
`.state` always retains the original server string. Unknown values never become
successful completion by default.

Event identity (`event_id`), run identity (`run_id`), sequence, and kind remain
separate. Core's `x-synth-event-kind` annotations tie a non-null objective payload
to its envelope kind. Wrong-kind payloads are refused instead of being selected
by their field shape. Null historical payloads remain readable.

`StreamCheckpoint.to_data()` returns `StreamCheckpointData`. Its JSON carries
`schema_version: 1`; restoration rejects unsupported versions and still reads the
original unversioned v1 shape. A cursor commits only at the final SSE frame of
an expanded sequence, while pending event IDs prevent duplicate delivery.

## Errors and migration

Catch `SwarmsAPIError` for server refusals and inspect its typed `problem`, status,
code, request ID and idempotency key. `ErrorKind` categorizes local transport,
contract, observation, integrity and lifecycle failures; local error classes
provide stable codes such as `wait_timeout` and `upload_incomplete`. Recovery
snapshots and checkpoints remain on the relevant exceptions.

Use attributes on `auth.status()` and stored credential records. Secrets are
excluded from record reprs. Use `model_dump(mode="json")` or `model_dump_json()`
when an actual JSON representation is needed.

## Checks

Run `uv sync --locked --extra mcp`, `uv run ty check`, `uv run ruff check synth_ai
scripts tests`, and `uv run ruff format --check synth_ai scripts tests`. CI also
checks both generators, regression tests, wheel metadata and an isolated wheel
installation. The wheel ships its PEP 561 marker.

[ty configuration](https://docs.astral.sh/ty/configuration/) is kept in
`pyproject.toml`; all enabled diagnostics fail the check. Ruff checks errors,
imports, bug patterns, modern syntax and Ruff-specific correctness rules. A
source regression check prevents explicit `Any` annotations from returning.
