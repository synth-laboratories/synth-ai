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
contents, Contribution lookup, and revision status require no API key. Supply
`SYNTH_API_KEY` through your authorized agent secret configuration only for
authenticated/private reads or explicitly enabled contribution tools. The
executable does not discover Index credentials from home files or Keychain.
Both Index flags accept only `true` or `false`. Missing Index opt-in means no
Index tools. Enabling Index requires an explicit backend URL; enabling writes
also requires a nonempty explicit key. Initialization and tool discovery
construct no SDK client and make no requests.

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

Unreleased typed Index client. Backend `packages/contributions/*` owns the wire
vocabulary and validators; this mirror must stay schema- and behavior-compatible.
Do not import backend modules from the published package. Cross-repo parity
checks live in `testing` (`test_index_openapi_parity`, clean-install script).

See `docs/drafts/synth-index-api-design-2026-09-12.md`. Fast search is the MVP;
deep execution handles are not implemented. No release is claimed by this code.

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
| `account.retrieve / contributions / usage / rewards / update_profile / update_pins` | caller identity, work, usage, credits, profile |
| `profiles.retrieve`, `rewards.award / reverse`, `contests.*` | public profiles; award/contest operator grants |

Every operation is declared once in `client.OPERATIONS` and executed identically
by sync and async clients. `test_index_openapi_parity` requires every SDK operation
to hit a real backend route with the same operation ID, and vice versa.

## Errors

Transport raises typed `SynthError` subclasses: `RateLimitedError` (with
`retry_after_seconds` from `Retry-After`), `PaymentRequiredError` (private cap or
wallet), `AuthorizationError` (including uninvited private scope), `ConflictError`,
`TransientServiceError`. `index_error_code(error)` returns the stable
`IndexErrorCode`. An unavailable service is never reported as an empty result.

## Upload

`contributions.upload(prepared, content)` (sync) and the async twin transfer an
explicit `{logical_path: bytes}` mapping, bounded to 64 MiB, after validating every
size, digest and target, using a credential-free storage client with no redirects.
They never read files, finalize, submit or retry. Retry by preparing again with the
same publication ID. The MCP `index_contribution_upload` tool reads only explicitly
listed files under an explicit root and rejects symlinks, escapes and credential-like
content before calling this path.
