# Synth Index SDK

For an API/MCP-only coding-agent integration, use the copyable
[setup prompt](AGENT_SETUP_PROMPT.md). It does not run a paid search or create a
product UI.

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

Replace the backend URL with the actual deployment URL. Anonymous public
browse (contents, Contribution lookup, and revision status) requires no API key.
Search, including public-scope FAST, requires `SYNTH_API_KEY` through your
authorized agent secret configuration. The
executable does not discover Index credentials from home files or Keychain.
Both Index flags accept only `true` or `false`. Missing Index opt-in means no
Index tools. Enabling Index requires an explicit backend URL; enabling writes
also requires a nonempty explicit key. Initialization and tool discovery
construct no SDK client and make no requests.

Read-only mode without a key exposes exact contents, Contribution lookup, and
revision status, but not search. With a key, it also exposes `index_search`,
`index_search_create`,
`index_search_get`, `index_search_result`, `index_search_events`, and
`index_search_cancel` for durable fast/deep Search recovery. Without a key,
those lifecycle tools are not advertised; the remaining tools use only the credential-free
`/api/v1/index/public/*` routes and reject private scope locally. With a key,
reads use the authenticated Index routes and may request authorized private
collections. Public-scope FAST search needs explicit wallet consent and a
charge ceiling of at least 5 cents. To deliberately contribute from this machine,
additionally set
`SYNTH_INDEX_MCP_WRITE_ENABLED=true`; this exposes draft creation, explicit
selected-file upload, and submission for review. Advanced Research tools do not
bypass this write gate. Credentials never belong in tool arguments. Index calls
use captured process configuration and own/close one SDK transport per
invocation. Every search is funded and charged according to backend account
policy. No MCP tool grants access, approves research, publishes Contributions,
or awards credits.

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

See `docs/drafts/synth-index-api-design-2026-09-12.md`. Fast search is the
bounded retrieval path. Deep search uses a durable server execution and never
silently falls back to fast search.

Grounded answers are a separate authenticated operation. Search continues to
return evidence; `answer(...)` performs fast or deep retrieval, fail-closed
evidence admission and cited synthesis under one explicit idempotency key:

```python
from uuid import uuid4

from synth_ai import SynthClient

with SynthClient() as synth:
    result = synth.index.answer(
        query="Why did the retrieval experiment reject launch readiness?",
        mode="fast",
        idempotency_key=str(uuid4()),
    )
    if result.status == "answered":
        print(result.answer, result.citations)
    else:
        print(result.insufficient_evidence_reason)
```

Every returned claim names exact digest-bound citation spans. Unsupported or
revoked evidence returns `insufficient_evidence`, never uncited prose. The
credential-free public client intentionally does not expose answer generation.

## Anonymous public browse and funded search

Browsing published Contributions needs no account or API key:

```python
from synth_ai.sdk.index import PublicIndexClient

with PublicIndexClient() as index:
    print(index.capabilities())
    print(index.tags.list())
```

Use `AsyncPublicIndexClient` with `async with` for native async applications.
Both clients own and close their HTTP transport. Search uses the authenticated
client, even when its scope is public. For FAST, explicitly consent to wallet
funding and bound the maximum charge; the organization must also have a valid
funding policy and sufficient balance:

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import SearchBillingConstraints

with SynthClient() as synth:  # SYNTH_API_KEY is required
    result = synth.index.search(
        query="RLVR verifier design",
        mode="fast",
        billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=5),
    )
    print(result.response, result.usage)
```

The 5-cent value is a caller ceiling for the current public FAST price, not a
claim that another mode or a future price is free. A declined or exhausted
funding source fails before a search result is delivered.

Deep search requires the authenticated client and a deployment whose
capabilities advertise deep mode. The convenience call waits for the same
durable Search identity through completion:

```python
from uuid import uuid4

from synth_ai import SynthClient
from synth_ai.sdk.index import SearchBillingConstraints

with SynthClient() as synth:
    result = synth.index.search(
        query="Compare the evidence for the two retrieval designs",
        mode="deep",
        billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=25),
        idempotency_key=str(uuid4()),
    )
    print(result.search_id, result.status)
```

Public or private DEEP requires a mode grant and funding. This wallet example
authorizes at most 25 cents; the backend may stop at that bound. The current
per-search ceiling is $1, and DEEP requires a ceiling of at least 10 cents.

For reconnect, progress, events, explicit cancellation, or a local wait timeout,
create the handle directly. A local timeout preserves `handle.search_id` and does
not cancel the server execution:

```python
from uuid import uuid4

from synth_ai.sdk.index import SearchBillingConstraints, SearchSpec

handle = synth.index.searches.create(
    SearchSpec(
        query="Trace the qualified evidence and identify unresolved questions",
        mode="deep",
        billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=25),
    ),
    idempotency_key=str(uuid4()),
)
result = handle.wait(timeout_seconds=30)
```

The CLI provides the same lifecycle through `synth-ai index searches create`,
`get`, `result`, `events`, and `cancel`. `create` requires an idempotency key and
prints the Search ID immediately. `events --after N` pages progress without
restarting the Search. `result` reads the saved Search specification first, so
the result is validated against the original request. These commands require
`SYNTH_API_KEY` or `--api-key`. An unfinished result returns the backend's
typed `index_search_result_not_ready` failure.

HTTP failures expose their stable Index code through `error.failure.code`,
request and correlation IDs when supplied by the backend, and a retry directive.
`index_error_code(error)` maps known values to `IndexErrorCode`; unknown future
codes remain available as raw strings. A failed durable execution records its
terminal `Search.failure.code` and `retryable` status in the Search snapshot.

The credential-free surface contains only public browse operations:

| Call | Route |
| --- | --- |
| `capabilities()` | `GET /public/capabilities` |
| `contents.retrieve(...)` | `POST /public/contents` |
| `tags.list()` | `GET /public/tags` |
| `contributions.retrieve(...)`, `contributions.revisions.retrieve(...)` | published Contribution and exact revision |
| `contributions.assets.retrieve(reference, asset_id)` | declared asset bytes of a published revision |
| `profiles.retrieve(principal_id)` | contributor profile as an anonymous reader sees it |

Each browse operation is the public twin of an authenticated operation. Search
is deliberately absent from `PublicIndexClient` and uses `SynthClient().index`.

## Surface (`SynthClient().index`, async twin on `AsyncSynthClient`)

| Call | Route |
| --- | --- |
| `search(...)`, `capabilities()` | `POST /search`, `GET /capabilities` |
| `answer(...)` | `POST /answer` cited fast/deep answer or explicit insufficient evidence |
| `searches.create / retrieve / result / events / cancel` | durable fast/deep execution lifecycle under `/searches` |
| `contents.retrieve(...)` | `POST /contents` |
| `contributions.create / retrieve / prepare_upload / upload / finalize / submit` | contributor workflow |
| `contributions.publish / withdraw` | `POST .../publication`, `.../withdrawal` (publisher grant / owner) |
| `contributions.revisions.create / retrieve`, `contributions.assessments.list` | revise; exact status, sealed package, citation |
| `reviews.list(status=)`, `contributions.reviews.create` | reviewer queue and decisions (never self-review) |
| `contributions.assets.retrieve(reference, asset_id)` | declared asset bytes |
| `tags.list()`, `collections.list()`, `collections.grants.*` | taxonomy; owner-only explicit shares |
| `account.retrieve / contributions / usage / promo_credit / rewards / update_profile / update_pins` | caller identity, work, usage, promo balance, credits, profile |
| `profiles.retrieve`, `rewards.award / reverse`, `contests.*` | public profiles; award/contest operator grants |

Every operation is declared once in `client.OPERATIONS` or
`client.PUBLIC_OPERATIONS` and executed identically by sync and async clients.
`test_index_openapi_parity` requires every SDK operation to hit a real backend
route with the same operation ID. Six backend operations are deliberately outside
the customer SDK: `index.contributions.research.lookup` is an operator acceptance
receipt lookup, and five `/operator/*` routes manage grants and billing
diagnostics.

## Customer usage accounting

The backend `search_funding` service owns funding and settlement. The accounting
models in `usage_accounting.py` project that authority; they do not quote prices,
grant access, infer consent, or derive charges from token or infrastructure cost.
Funding values are `none`, `promo_credit`, `deep_beta`, and `wallet` (or null when
no funding fact exists). `index_deep_beta` is not a wire value.

`CustomerCharge` adds `funding_source`, `terminal_outcome`, and
`adjustment_microcents` to the existing
currency, price, reservation, settlement, release, refund, and ledger fields.
`SearchUsageSummaryRow` adds `price_version`, `funding_source`, `terminal_outcome`,
`settlement_state`, `reserved_microcents`, `released_microcents`, and
`refunded_microcents`, and `adjustment_microcents`. Existing consumption counters
and pagination remain.
Terminal outcomes use `SearchSettlementOutcome`, including `complete` and
`insufficient_evidence`; `grounded` is not an accounting outcome.

Receipt `settled_microcents` and summary/CSV `customer_charge_microcents` carry
the backend's corrected settlement totals. Consumers must not add corrections
again or subtract the separately reported refunds a second time. Reserved and
released values describe reservation history, not the live wallet hold balance.
Microcents convert to USD by dividing by 100,000,000.

Correction totals are signed and informational: outstanding refunds equal the
stored original refund baseline minus the sum of signed adjustments; customer
settlement equals immutable gross settlement minus outstanding refunds.
Net settlement plus refunds plus releases cannot exceed the original reservation.
Neither refunds nor positive reversals reopen beta units or reset gross caps.
Summary periods follow original settlement creation (first observed usage for
searches without settlement), so later corrections remain in the original cohort.

Settlement-backed receipts exist even without physical events: their observation
arrays are empty, recorded counts are zero, and measurement state is `pending`.
Summary rows add `measurement_state` and `unmeasured_search_count`; consumption
is null for groups with unmeasured searches. A missing physical write never erases
financial facts or invents measured zero usage. Server mode sources must agree
before one Search contributes one charge to a summary.

Customer receipt and summary models omit infrastructure and unallocated costs.
The backend must also filter internal cost metrics from customer operation totals
and CSV exports; the shared metric vocabulary does not authorize disclosure.

The vendored `openapi/index-v1.json` is copied from the selected backend's
router-generated `contracts/synth_index_openapi.json`. The SDK mirrors the
backend's cited-only Search delivery, 180-second default / 300-second maximum
durable Deep execution, and 1,024-operation receipt bound. Synchronous
`POST /search` remains limited to a 90-second Deep wait; the SDK's Deep
convenience path uses durable create/wait instead. Source and schema parity do
not replace an installed-wheel test against the matching deployed backend.

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
target because it expired, intake prepares again once and transfers only what is
still missing.

State (`synth.index.research-intake-state.v2`) names the backend, organization
and account it was allocated under, and is refused against any other — an
allocated draft and its idempotency keys mean nothing there. A `.lock` sidecar
holds the state file for one process at a time and names its holder, so a stale
lock is cleared deliberately. Corrupt or foreign state is reported with what to
do about it; it is never silently discarded.
