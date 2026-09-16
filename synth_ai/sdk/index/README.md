# Synth Index SDK

## External agents over stdio MCP

Install the reviewed SDK package, then configure your agent's MCP server:

```json
{
  "mcpServers": {
    "synth-index": {
      "command": "synth-ai-research-mcp",
      "env": {
        "SYNTH_INDEX_MCP_ENABLED": "true",
        "SYNTH_INDEX_MCP_WRITE_ENABLED": "false",
        "SYNTH_BACKEND_URL": "https://your-configured-synth-backend.example"
      }
    }
  }
}
```

Replace the backend URL with the actual deployment URL. Public read-only search,
contents, Contribution lookup, revision status, asset bytes, capabilities and
tags require no API key. Supply `SYNTH_API_KEY` or `SYNTH_API_KEY_FILE` through
your authorized agent secret configuration only for authenticated/private reads,
account reads (identity, own Contributions, usage, promo balance) or explicitly
enabled contribution tools. The
executable does not discover Index credentials from home files or Keychain.
Both Index flags accept only `true` or `false`. Missing Index opt-in means no
Index tools. Enabling Index requires an explicit backend URL; enabling writes
also requires a key (`SYNTH_API_KEY` or `SYNTH_API_KEY_FILE`, not both).
Initialization and tool discovery construct no SDK client, read no key file and
make no requests.

### Long-running sessions: key rotation and revocation

`SYNTH_API_KEY` is captured once at start, so an agent cannot change a running
server's identity. For sessions that outlive a key, set `SYNTH_API_KEY_FILE` to
a mode-600 regular file holding the key: it is read on every tool invocation, so
replacing its content rotates the key without a restart. When the backend
answers 401, the tool re-reads the file once; a different key reruns the tool
(the rejected request never executed), an unchanged key raises
`CredentialRevokedError`. A 403 is a permission decision and is returned as is.
The CLI behaves the same with `--api-key-file` / `SYNTH_API_KEY_FILE`.

Read-only mode exposes search, exact contents, Contribution lookup, and revision
status. Without a key, those tools use only the credential-free
`/api/v1/index/public/*` routes and reject private scope locally. With a key,
reads use the authenticated Index routes and may request authorized private
collections. To deliberately contribute from this machine, additionally set
`SYNTH_INDEX_MCP_WRITE_ENABLED=true`; this exposes draft creation, explicit
selected-file upload, and submission for review. Advanced Research tools do not
bypass this write gate. Credentials never belong in tool arguments. Index calls
use captured process configuration and own/close one SDK transport per
invocation. Public search is free under backend rate limits; requesting private
collections can be charged according to backend account policy. No MCP tool
grants access, approves research, publishes Contributions, or awards credits.

Local MCP file upload requires POSIX descriptor-relative, no-follow file access
and bounds actual bytes read; it refuses unsupported platforms. Windows users
can use the SDK's explicit-bytes upload API instead. No directories are scraped
or uploaded implicitly.

Agent retrieval opt-in is read-only. `IndexAccessPolicy.allow_draft_preparation`
and Intern `index_draft_enabled` default to false and require reads enabled.
Draft preparation needs explicit persisted operator intent as well as an
effective capability grant; metadata cannot authorize it. Draft opt-in never
authorizes submission, approval, publication, or private-search charges.

The typed Index client mirrors backend `packages/contributions/*`, which owns the wire
vocabulary and validators; this mirror must stay schema- and behavior-compatible.
Do not import backend modules from the published package. Cross-repo parity
checks live in `testing` (`test_index_openapi_parity`, clean-install script).

See `docs/drafts/synth-index-api-design-2026-09-12.md`. Fast search is the MVP;
deep execution handles are not implemented.

## Free public search

Public search needs no account or API key and cannot select private collections:

```python
from synth_ai.sdk.index import PublicIndexClient

with PublicIndexClient() as index:
    result = index.search(query="RLVR verifier design", max_results=5)
    for hit in result.results:
        print(hit.title, hit.url)
```

Use `AsyncPublicIndexClient` with `async with` for native async applications.
Both clients own and close their HTTP transport. Authenticated and private
operations remain under `SynthClient().index`.

The credential-free surface is the complete set of eight customer operations
that need no account at all:

| Call | Route |
| --- | --- |
| `search(...)`, `capabilities()` | `POST /public/search`, `GET /public/capabilities` |
| `contents.retrieve(...)` | `POST /public/contents` |
| `tags.list()` | `GET /public/tags` |
| `contributions.retrieve(...)`, `contributions.revisions.retrieve(...)` | published Contribution and exact revision |
| `contributions.assets.retrieve(reference, asset_id)` | declared asset bytes of a published revision |
| `profiles.retrieve(principal_id)` | contributor profile as an anonymous reader sees it |

Each one is the public twin of an authenticated operation, so code written
against `SynthClient().index` reads the same against `PublicIndexClient`.

## Surface (`SynthClient().index`, async twin on `AsyncSynthClient`)

| Call | Route |
| --- | --- |
| `search(...)`, `capabilities()` | `POST /search`, `GET /capabilities` |
| `contents.retrieve(...)` | `POST /contents` |
| `contributions.create / retrieve / prepare_upload / upload / finalize / submit` | contributor workflow |
| `contributions.publish / withdraw` | `POST .../publication`, `.../withdrawal` (publisher grant / owner) |
| `contributions.revisions.create / retrieve`, `contributions.assessments.list` | revise; exact status, sealed package, citation |
| `reviews.list(status=)`, `contributions.reviews.create` | reviewer queue and decisions (never self-review) |
| `contributions.assets.retrieve(reference, asset_id)` | declared asset bytes |
| `tags.list()`, `collections.list()`, `collections.grants.*` | taxonomy; owner-only explicit shares |
| `account.retrieve / contributions / usage / promo_credit / rewards / update_profile / update_pins` | caller identity, work, usage, promo balance, credits, profile |
| `profiles.retrieve`, `rewards.award / reverse`, `contests.*` | public profiles; award/contest operator grants |

| `contributions.create_research / lookup_research` | private research allocation and its non-mutating recovery read |

Every operation is declared once in `client.OPERATIONS` or
`client.PUBLIC_OPERATIONS` and executed identically by sync and async clients.
The typed clients cover all 48 backend operations. The release parity gate
(`testing/backend/test_index_operation_parity.py`) checks each one in both
directions against the backend contract: method and path; path, query and
header parameters (the required `Idempotency-Key` is always sent, the operator
`Synth-Acceptance-*` headers never are); request and response models field by
field, recursively; byte responses; and that the sync and async clients send
identical requests.

### CLI and MCP coverage

`synth_ai.sdk.index.surfaces` declares which operations each `synth-ai index`
command and each MCP tool sends, with and without a key, and the reason for
every operation a surface leaves out (reviewer, publisher, sharing, reward,
contest and profile-editing workflows stay in the typed client and web app).
`test_index_surface_parity.py` fails when a backend operation has no decision on
a surface or a command/tool sends something it does not declare.

| CLI (`synth-ai index ...`) | MCP tool |
| --- | --- |
| `capabilities`, `tags` | `index_capabilities`, `index_list_tags` |
| `search [--private]` | `index_search` |
| `contents`, `contribution`, `revision` | `index_get_contents`, `index_get_contribution`, `index_contribution_status` |
| `asset --output FILE` | `index_get_asset` (at most 1 MiB, base64) |
| `account`, `my-contributions`, `usage`, `promo-credit` | `index_account`, `index_my_contributions`, `index_usage`, `index_promo_credit` |
| `research preview / submit / recover-state` | — (intake reads local files and keeps local state) |
| — | `index_contribution_create / upload / submit` (write opt-in) |

## Errors

Transport raises typed `SynthError` subclasses: `RateLimitedError` (with
`retry_after_seconds` from `Retry-After`), `PaymentRequiredError` (private cap or
wallet), `AuthorizationError` (including uninvited private scope), `ConflictError`,
`TransientServiceError`. `index_error_code(error)` returns the stable
`IndexErrorCode`. An unavailable service is never reported as an empty result.
`IndexErrorCode` names every code an Index route can return; the two
acceptance-run codes a customer credential cannot provoke are listed in
`OPERATOR_ONLY_ERROR_CODES`. `test_index_error_parity.py` scans the backend
source and fails on any code the SDK does not name, or names but the backend
never returns. Unknown future codes stay available as `error.failure.code`.

## Upload

`contributions.upload(prepared, content)` (sync) and the async twin transfer an
explicit `{logical_path: bytes}` mapping, bounded to 64 MiB, after validating every
size, digest and target, using a credential-free storage client with no redirects.
They never read files, finalize, submit or retry. Retry by preparing again with the
same publication ID. The MCP `index_contribution_upload` tool reads only explicitly
listed files under an explicit root and rejects symlinks, escapes and credential-like
content before calling this path.

## Research bundle intake

`synth-ai index research preview <conversion-dir>` verifies the converted
`package/` bytes and the offline receipt, then shows source, evidence, private
audience, pending qualification and unattested rights. It makes no API call.
The backend research-bundle converter must produce this directory from a sealed
`synth.research.export-bundle.v1`; this SDK does not parse raw sessions.

`synth-ai index research submit <conversion-dir>` allocates a private SYNTH-origin
draft through `POST /index/contributions/research`, rebinds only its server-issued
Contribution/revision IDs, prepares exact bytes, streams the requested objects,
finalizes them, and submits the private revision for review. The server can keep
rights pending while barring any org/public release. Use `--finalize-only` to stop
after upload when an arc must be held before review. The backend enforces its
mandatory submission and release gates in either case.
The backend requires an active `research_import`
grant and independently rejects prohibited REB source paths. Re-running the command
reuses the same draft key and publication ID in `.research-intake-state.json`,
obtains fresh targets and lets prepare omit already-present objects. The state file
contains no API key or signed upload URL. Intake never sets rights attestation,
qualifies, publishes, or broadens the private audience.

Resuming is answered by the server, not by the saved file. Each run reconciles
the saved reference against the current revision before acting, so a mutation
whose response was lost is recovered rather than repeated; a revision the server
has already advanced to `qualified` or `published` is reported as it stands; and
a `rejected`, `withdrawn` or `changes_requested` revision raises
`TerminalRevision` instead of being submitted again. If storage refuses a signed
target because it expired, intake prepares again (up to three times per run) and
transfers only what is still missing; re-running continues from there.

Every mutation is protected against a lost response, in the same run and in a
later one:

- allocation: the state records that the key was sent *before* sending it. If
  the response is lost, intake asks
  `POST /index/contributions/research/lookup` (`contributions.lookup_research`),
  a read of the caller's own receipt for that key; it allocates only when the
  server answers `research_receipt_absent`. The allocation is never replayed.
- finalize and submit: a failure is followed by one revision read; if the
  server committed, intake carries on. If that read also fails, the original
  error is raised and the next run reconciles first.

State (`synth.index.research-intake-state.v2`) names the backend, organization
and account it was allocated under, and is refused against any other — an
allocated draft and its idempotency keys mean nothing there. An operating-system
lock on a `.lock` sidecar holds the state file for one process at a time; it is
released automatically if the process dies, and the sidecar names the current
holder. State is written with a same-directory temporary file, `fsync` and an
atomic rename. Corrupt, linked or unknown-schema state is reported with what to
do about it; it is never silently discarded, and nothing is sent.

### Recovering v1 saved state

Intake state written by earlier development builds
(`synth.index.research-intake-state.v1`) recorded the draft key and publication
ID but not the backend or account. Current intake refuses to resume it, and
deleting it is unsafe: a fresh state file would allocate a second draft if the
first run had already reached the server. Convert it instead, with the account
and backend the old run used:

```bash
synth-ai index research recover-state <conversion-dir> \
  --backend-url https://<the backend the old run used> \
  --api-key-file ~/.config/synth/index-key   # or SYNTH_API_KEY
# then resume exactly as before:
synth-ai index research submit <conversion-dir> --backend-url ... --api-key-file ...
```

`recover-state` (`research_intake.recover_v1_state`) never allocates, uploads,
finalizes or submits. It reads the server's receipt for the saved key:

| Server answer | Result |
| --- | --- |
| a draft under this key | v2 state bound to that draft, its server status (`finalized`/`submitted` carried forward; terminal statuses stay terminal on resume) |
| nothing allocated, and v1 state recorded no server work | v2 state with the same key and publication ID, marked so the next run looks before allocating |
| nothing allocated, but v1 state names a draft or progress | refused, nothing changed: the state belongs to another backend or account |
| key used for different input, lookup forbidden, or still committing | refused, nothing changed |

The original file is kept beside the new one as `<state>.v1-backup`. Running
`recover-state` again on converted state reports `recovered: false`.
