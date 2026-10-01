# Synth Index SDK

Two Search routes, chosen explicitly by the client (never by the presence of a
key):

- **Public route** (`PublicIndexClient().public_search`, CLI
  `synth-ai index search --public`, MCP `index_search` without a key): anonymous
  only, free to the caller, published public Contributions only. Limits and
  retention are published by the backend in `capabilities().public_search`.
  Available only where the backend enables it.
- **Keyed route** (`SynthClient().index.search`, CLI `--keyed`, MCP
  `index_search` with a key): paid and org-funded even when the selected corpus
  is public. Fast is 5 cents per delivered Search; Deep is 5 cents plus measured
  model cost, capped by your `max_charge_cents` and at $1 per Search. Funding is
  Index promo credit, then the wallet with explicit consent.

The public route refuses credentials (409), so having a key never makes a
request free, and a keyed request is never silently sent to the public route.

For an API/MCP-only coding-agent integration, use the copyable
[setup prompt](AGENT_SETUP_PROMPT.md). It does not run a paid search or create a
product UI.

## External agents over stdio MCP

Install the reviewed SDK package, then configure your agent's MCP server:

```json
{
  "mcpServers": {
    "synth-index": {
      "command": "synth-ai-index-mcp",
      "env": {
        "SYNTH_INDEX_MCP_WRITE_ENABLED": "false",
        "SYNTH_BACKEND_URL": "https://your-configured-synth-backend.example"
      }
    }
  }
}
```

The dedicated entrypoint advertises only Index tools; it does not advertise
unrelated Managed Research tools. The general `synth-ai-research-mcp` remains
available and requires `SYNTH_INDEX_MCP_ENABLED=true` to add Index tools.
Replace the backend URL with the actual deployment URL. Anonymous public
browse (contents, Contribution lookup, and revision status) and the public
search route (`index_search`, Index Search v0.2) require no API key. Private
and durable Search tools require `SYNTH_API_KEY` through your authorized agent
secret configuration. The
executable does not discover Index credentials from home files or Keychain.
Index flags accept only `true` or `false`. The dedicated entrypoint enables
Index without a flag and requires an explicit backend URL; enabling writes
also requires a nonempty explicit key. Initialization and tool discovery
construct no SDK client and make no requests.

Read-only mode without a key exposes exact contents, Contribution lookup,
revision status, and `index_search`: the public fast/deep search over
`POST /api/v1/index/public/search`. Its price, rate limits, retention and
privacy line are read from `GET /api/v1/index/capabilities` (`public_search`)
at call time and returned under `terms`; the tool text never states a price
by hand, and the tool fails closed when the backend has the route disabled
or is unavailable (429 errors carry `retry_after_seconds` and the limit
scope). Public results report `route: "public"`. With a key, `index_search` is a paid search charged to your
organization's wallet (`POST /api/v1/index/search`). It runs only when the
organization has turned on wallet payments for that mode: the tool reads
`GET /api/v1/index/me/access-funding` (cached for a minute), sends an explicit
per-call ceiling (Fast: the published Fast price; Deep: the docs ceiling or the
organization's monthly limit, whichever is lower) and returns the `charge`
(amount, wallet debit, receipt) with `route: "keyed"` and `paid: true`. Keyed
results carry no `terms`: the public terms do not describe keyed Search, and
keyed price/funding come from your organization's access-funding policy.
Without consent it refuses with
`index_wallet_consent_required` and the steps to enable it; no paid request is
sent and nothing is charged. With a key, it also exposes `index_private_search`,
`index_search_create`,
`index_search_get`, `index_search_result`, `index_search_events`, and
`index_search_cancel` for funded private search and durable Search recovery.
Without a key, those tools are not advertised; the remaining tools use only
the credential-free `/api/v1/index/public/*` routes and reject private scope
locally. With a key, reads use the authenticated Index routes and may request
authorized private collections. Private or wallet-funded search needs explicit
wallet consent and a charge ceiling at or above the price published in
capabilities. To deliberately contribute from this machine,
additionally set
`SYNTH_INDEX_MCP_WRITE_ENABLED=true`; this exposes draft creation, explicit
selected-file upload, and submission for review. Advanced Research tools do not
bypass this write gate. Credentials never belong in tool arguments. Index calls
use captured process configuration and own/close one SDK transport per
invocation. Every keyed search is funded and charged according to backend
account policy; the anonymous public route is free to the caller. No MCP tool
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

See `docs/drafts/synth-index-api-design-2026-09-12.md`. Fast search is the
bounded retrieval path. Deep search uses a durable server execution and never
silently falls back to fast search.

## Public search (Index Search v0.2)

Public fast and deep search over reviewed Contributions works with no account
or API key, and only without one: the public route is anonymous-only and refuses
any credentialed request with `PublicSearchAuthenticatedError` (409
`index_public_search_authenticated`). Keyed callers use the paid
`IndexAPI.search(...)`. The backend publishes
the price, rate limits, retention and privacy wording in
`capabilities().public_search`; read them from there rather than hardcoding:

```python
from synth_ai.sdk.index import PublicIndexClient, PublicSearchRateLimitedError

with PublicIndexClient() as index:
    terms = index.public_search_terms()  # price / limits / privacy sentences
    print(terms.price, terms.limits, terms.privacy)
    try:
        result = index.public_search("RLVR verifier design", mode="fast", max_results=5)
    except PublicSearchRateLimitedError as error:
        print(error.scope, error.retry_after_s)
    else:
        print(result.search_id, result.customer_charge_cents, result.monitor_release_id)
        print(result.response)  # claims cite contribution ids inline: [<contribution_id>]
        for item in result.citations:  # first-appearance order; fetch content by id
            print(item.contribution_id, item.revision_id)
```

`result.monitor_release_id` is the Monitor's release decision id. The delivered
body is exactly the Monitor-reviewed public contract (`request_id`, `mode`,
`status`, the four version fields, `response`, `citations`, `amount_cents`), so
every public-only field travels in a response header instead: the release id in
`X-Index-Monitor-Release`, the Search id in `X-Index-Search-Id`, the token in
`X-Search-Token` / `X-Search-Token-Expires-At` and the zero charge in
`X-Index-Customer-Charge-Cents`. The SDK reads the headers first and falls back
to the older body fields (`search_id`, `usage`, `monitor.release_id`).

`mode="deep"` is admitted with 202 and polled at the backend's cadence until
delivered (default wait `DEFAULT_PUBLIC_SEARCH_WAIT_SECONDS`). Pass
`wait=False` to receive a `PublicSearchHandle` with `poll()`, `wait()`,
`replay()` and `cancel()`; the handle holds the per-search token and never
prints it. Reconnect later with `index.public_search_handle(search_id, token)`.
Errors are typed per backend code: `PublicSearchRateLimitedError` (429, with
`retry_after_s` and `scope`), `PublicSearchBudgetExhaustedError`,
`PublicSearchRateStoreUnavailableError`, `PublicSearchMonitorUnavailableError`
(each 503 fails closed, nothing charged), `PublicSearchRequestTooLargeError`
(413), `PublicSearchDisabledError` (404 when the backend flag is off) and
`PublicSearchNotFoundError` (404 for a wrong token). On `SynthClient().index`
the API key is attached, so `public_search` raises `PublicSearchAuthenticatedError`;
call `search(...)` (paid, per-org limits) there. The MCP `index_search` tool never
sends a key to the public route: with a key it runs the paid search only when the
organization has opted in to wallet funding for the mode, and otherwise raises
`WalletConsentRequiredError` (`index_wallet_consent_required`).

From the CLI (0.21.1+), `synth-ai index search QUERY --public [--mode deep]`
runs the same anonymous route and prints the result with the backend's `terms`.
It never sends a key, even when `SYNTH_API_KEY` is inherited from the
environment, and it refuses keyed-only options (`--api-key`,
`--private-collection`, `--allow-wallet`, `--max-charge-cents`,
`--deadline-seconds`). See [Errors](#errors) for the public-route refusals.

## Anonymous public browse and keyed (paid) search

### Contribution keyword filters

Keywords use the existing tag registry (`index.tags.list()`), including aliases
resolved to canonical tag IDs by the backend. The Search pins the registry
version used to resolve its filters. `SearchFilters.tags_any` requires at least one selected
tag; `tags_all` requires every selected tag; `tags_none` excludes any selected
tag. The three constraints combine with AND and apply before retrieval in both
FAST and DEEP. Each list accepts at most ten distinct identifiers. Contradictory
inclusion/exclusion is rejected. Unknown or inactive tags are rejected by the
backend before execution.

```python
from synth_ai.sdk.index import SearchFilters

# Add this argument to keyed search(...) or anonymous public_search(...).
filters = SearchFilters(tags_any=("optimization",), tags_none=("obsolete",))
```

The identifiers above are examples; select active identifiers from the registry.
The CLI mirrors these fields with repeatable `--tag-any`, `--tag-all`, and
`--tag-none` options on `index search` and `index searches create`. MCP
`index_search` accepts `filters`; `index_private_search` and
`index_search_create` accept the same filters inside their `search` intent.

Restricted testing material requires an authenticated organization with a live
operator grant, existing Artifact read authorization, explicit private collection
selection, and positive selection of its required keyword:

```python
from synth_ai.sdk.index import SearchBillingConstraints, SearchSpec
from synth_ai.sdk.index.search import SearchScope

intent = SearchSpec(
    query="Compare the Cybernetics evidence",
    scope=SearchScope(visibility="private", collection_ids=(testing_collection_id,)),
    filters=SearchFilters(tags_all=("cybernetics",)),
    billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=charge_ceiling),
)
result = synth.index.search(intent, idempotency_key=saved_request_key)
```

`testing_collection_id`, `charge_ceiling`, and `saved_request_key` come from your
authorized collection and funding configuration. The query text alone does not
opt in; the required keyword must be in `tags_any` or `tags_all`. Keywords never
grant access. Operator restrictions survive author tag edits, and reads recheck
current access. Restricted testing material is excluded from public search and
public discovery. Removing an operator grant revokes access. Keyword selection
and reader MCP tools cannot mint these grants. Frozen experiment membership remains a separate
server-side concern; mutable keyword selection does not freeze an experiment.

Operators with the dedicated `testing_corpus_admin` capability configure a
testing corpus independently of author-editable tags. The backend routes are:

| Operation | Route (under `/api/v1/index`) |
| --- | --- |
| Create corpus | `POST /testing-corpora` with `corpus_key`, `required_tag`, `reason` |
| Bind exact sealed revision | `POST /testing-corpora/{corpus_id}/contributions/{contribution_id}/revisions/{revision_id}` with `reason` |
| Grant testing organization | `POST /testing-corpora/{corpus_id}/organizations/{org_id}/grants` with `reason`, optional `expires_at` |
| Revoke organization grant | `POST /testing-corpora/grants/{grant_id}/revoke` with `reason` |
| Disable corpus | `POST /testing-corpora/{corpus_id}/disable` with `reason` |

Membership binds an exact Contribution revision in a committed private/org
Artifact collection. Operators separately provision Artifact read access and
provide the selected collection IDs to testing organizations. A testing grant
does not replace Artifact authorization or the explicit collection selection.
Registry keywords are metadata; the operator policy enforces isolation even if
an author changes or removes a keyword. Corpus disabling, expired grants and
revocations are checked again when reading a saved Search.

For restricted citations, reuse the delivered Search ID on subsequent reads:

```python
reference = result.citations[0]
revision = synth.index.contributions.revisions.retrieve(reference, search_id=result.search_id)
contents = synth.index.contents.retrieve(references=(reference,), search_id=result.search_id)
# The same optional search_id is accepted by contributions.retrieve(),
# contributions.assessments.list(), and contributions.assets.retrieve().
```

The receipt is required for testing-corpus readers; it grants no authority by
itself. Existing reads omit the optional query parameter. MCP
`index_get_contribution` and `index_contribution_status` accept `search_id` for
authenticated readers; `index_get_contents` already accepts it. The CLI exposes
the receipt as `index contribution status ... --search-id ID`. Raw Artifact
manifest/asset URLs also accept `?search_id=ID`, under their existing authorization.

Reading public capabilities, tags, and a known published Contribution ID needs
no account or API key:

```python
from synth_ai.sdk.index import PublicIndexClient

with PublicIndexClient() as index:
    print(index.capabilities())
    print(index.tags.list())
```

Use `AsyncPublicIndexClient` with `async with` for native async applications.
Both clients own and close their HTTP transport (public Search is on the
synchronous `PublicIndexClient`). Keyed search uses the authenticated `search()`
path and is paid whether the selected corpus is public or private. Explicitly
consent to wallet funding and bound the maximum charge; the organization must
also have a valid funding policy and sufficient balance:

```python
from uuid import uuid4

from synth_ai import SynthClient
from synth_ai.sdk.index import SearchBillingConstraints

request_key = str(uuid4())  # Save this with your job before sending the request.
with SynthClient() as synth:  # SYNTH_API_KEY is required
    result = synth.index.search(
        query="RLVR verifier design",
        mode="fast",
        billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=5),
        idempotency_key=request_key,
    )
    print(result.response, result.usage)
```

The ceiling is a caller bound; the backend publishes the Fast price as
`capabilities().private_search.price_cents_per_search` (keyed public-scope Fast
uses the same price). A declined or exhausted funding source fails before a
search result is delivered.

Keyed Deep search requires the authenticated client and a deployment whose
capabilities advertise deep mode (anonymous public Deep is
`public_search(..., mode="deep")` above). The convenience call waits for the same
durable Search identity through completion:

```python
from uuid import uuid4

from synth_ai import SynthClient
from synth_ai.sdk.index import SearchBillingConstraints

request_key = str(uuid4())  # Save this with your job before sending the request.
with SynthClient() as synth:
    result = synth.index.search(
        query="Compare the evidence for the two retrieval designs",
        mode="deep",
        billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=25),
        idempotency_key=request_key,
    )
    print(result.search_id, result.status)
```

Funded DEEP requires a mode grant and funding. It is charged 5 cents plus the
measured model cost, rounded up to a cent, never above your `max_charge_cents`
or $1 per Search; the ceiling must leave room for at least 5 cents of usage above
the base. This wallet example authorizes at most 25 cents; the backend may stop
at that bound.

For reconnect, progress, events, explicit cancellation, or a local wait timeout,
create the handle directly. A local timeout preserves `handle.search_id` and does
not cancel the server execution:

```python
from uuid import uuid4

from synth_ai import SynthClient
from synth_ai.sdk.index import SearchBillingConstraints, SearchSpec

request_key = str(uuid4())  # Save this with your job before sending the request.
with SynthClient() as synth:
    handle = synth.index.searches.create(
        SearchSpec(
            query="Trace the qualified evidence and identify unresolved questions",
            mode="deep",
            billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=25),
        ),
        idempotency_key=request_key,
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

For wallet-funded keyed CLI searches, provide explicit consent and a retail ceiling:

```sh
synth-ai index search "RLVR verifier design" --keyed --allow-wallet --max-charge-cents 5 --idempotency-key YOUR_UNIQUE_REQUEST_ID
synth-ai index searches create "Compare the retrieval designs" --mode deep --allow-wallet --max-charge-cents 25 --idempotency-key YOUR_OTHER_UNIQUE_REQUEST_ID
```

Omitting `--allow-wallet` and `--max-charge-cents` does not grant wallet
consent; an already-funded promo policy may still apply. `index search` with an
inherited `SYNTH_API_KEY` and no route flag uses the keyed route and says so on
stderr; with no key and no flag it refuses rather than guessing a route. Durable Deep CLI
commands accept the backend's 300-second execution maximum.

HTTP failures expose their stable Index code through `error.failure.code`,
request and correlation IDs when supplied by the backend, and a retry directive.
`index_error_code(error)` maps known values to `IndexErrorCode`; unknown future
codes remain available as raw strings. A failed durable execution records its
terminal `Search.failure.code` and `retryable` status in the Search snapshot.

The credential-free surface (`PublicIndexClient`) contains public known-ID and
catalog-metadata reads plus free public Search:

| Call | Route |
| --- | --- |
| `public_search(...)`, `public_search_handle(...)` | `POST /public/search`, `GET /public/searches/{id}` (sync client only) |
| `public_search_terms()`, `public_search_capability()` | `public_search` block of `GET /public/capabilities` |
| `capabilities()` | `GET /public/capabilities` |
| `contents.retrieve(...)` | `POST /public/contents` |
| `tags.list()` | `GET /public/tags` |
| `contributions.retrieve(...)`, `contributions.revisions.retrieve(...)` | published Contribution and exact revision |
| `contributions.assets.retrieve(reference, asset_id)` | declared asset bytes of a published revision |
| `profiles.retrieve(principal_id)` | contributor profile as an anonymous reader sees it |

Each browse operation is the public twin of an authenticated operation. Keyed
(paid) and private Search use `SynthClient().index`.

## Surface (`SynthClient().index`, async twin on `AsyncSynthClient`)

| Call | Route |
| --- | --- |
| `search(...)`, `capabilities()` | `POST /search`, `GET /capabilities` |
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

The contribution, profile, reward and contest operations are API reference for
authorized grants. Contributions, profiles, clout, credits and contests are not
part of the current Index release.

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
Funding values are `none`, `promo_credit`, `deep_beta`, `wallet`, and `service_free_public` (or null when
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

## Contribution journey: upload, review, repair, appeal, publish

What exists today, and where it stops. This section describes SDK, CLI and local
MCP behavior only. It makes no claim about a deployed backend: the private QA
routes are rollout-gated (an unavailable route is a typed error, never a local
fallback), and nothing here has been exercised against a real hosted backend,
Clerk tenant or native MCP client. The hosted MCP registry has no QA tools yet.

Requirements: Python 3.11+. The local stdio MCP server speaks protocol versions
`2025-06-18` and `2024-11-05`. No mobile client is qualified; do not assume the
flow works from one.

### Connect

| Path | Credential | Notes |
| --- | --- | --- |
| CLI `synth-ai index ...` | `--api-key` / `SYNTH_API_KEY`, `--backend-url` / `SYNTH_BACKEND_URL` | Both required, explicit, never read from disk or Keychain. |
| Python | `SynthClient().index`, or `index_with_oauth(base_url, access_token)` | `index_with_oauth` takes a token your agent already holds (HTTPS or loopback only). The SDK does no login, refresh or token storage. |
| Local MCP `synth-ai-index-mcp` | `SYNTH_API_KEY`; `SYNTH_INDEX_MCP_WRITE_ENABLED=true` for any mutating tool | Credentials never appear in tool arguments. |

### Flow

Each step is a separate, explicit action. Approval in QA is private acceptance
only; it is not publication, certification or a reward.

1. **Draft and upload.** `index contribution create --idempotency-key K`, then
   `index contribution upload DRAFT.json UPLOAD.json --root DIR --file logical=path`
   (only the listed files under `--root`; symlinks and escapes are refused), then
   `index contribution submit CONTRIBUTION REVISION SPEC.json`. MCP:
   `index_contribution_create`, `index_contribution_upload`, `index_contribution_submit`.
2. **Open a QA case** for the exact submitted revision: `index qa open SPEC.json`
   (MCP `index_qa_case_create`). The case pins the revision, manifest digest and rubric version.
3. **Review.** A coordinator invites a reviewer (`index qa invite`, `index_qa_invite_reviewer`);
   the reviewer accepts (`index qa accept`), reads the sealed package
   (`index_qa_package`, `index_qa_asset`; needs an accepted assignment), records checks
   and a review (`index qa review`, `index_qa_review_record`) and may request changes or
   approve through `index qa send` / `index_qa_reviewer_event`. A reviewer cannot
   self-approve their own Contribution.
4. **Respond and repair.** The contributor reads shared events (`index qa events`) and
   responds (`index qa send --action respond`). To repair, open a private child revision
   (`index contribution repair CONTRIBUTION PARENT_REVISION`, MCP `index_contribution_revise`),
   upload and submit it; the parent stays immutable and the child needs fresh review.
5. **Appeal or escalate.** `index qa appeal` (a rejection or private acceptance) and
   `index qa escalate` are fenced: they need `--expected-version`, `--manifest-digest`,
   `--rubric-version` and an idempotency key. A stale fence exits 5 and nothing is merged.
   A coordinator resolves with `index qa adjudicate`; the only outcome is reopening fresh
   independent review, never approval. `index qa note` adds an internal note the
   contributor never sees.
6. **Publish.** A separate publisher grant is required (`index contribution publish`,
   MCP `index_contribution_publish`), plus rights and author consent enforced by the
   backend. QA acceptance never grants it. `index contribution withdraw` removes a
   Contribution from Search and new reads; prior downloads cannot be recalled.

Retry rule: reuse the SAME idempotency key after any uncertain response (exit 6).

### Permissions

Scopes gate the class of operation. The backend decides who may act on which case
(ownership, assignment, coordinator role, organization) on every request, so holding a
scope never bypasses those checks. Any-of; a token with any listed scope reaches the class.
The SDK table is `synth_ai.sdk.index.scopes.OPERATION_SCOPES`, advertised as each local MCP
tool's `requiredScopes` and in `--help`; a test asserts it equals the backend table.

| Operation | Any of |
| --- | --- |
| Create draft, revise, upload, finalize, submit | `index:intake` |
| Create QA case | `index:intake`, `index:coordinate` |
| Read case, read shared events, post events, appeal, escalate | `index:intake`, `index:review`, `index:coordinate` |
| Internal note, record review/check, preflight, secret scan, list reviews/checks, list/accept assignments | `index:review`, `index:coordinate` |
| Invite or revoke a reviewer assignment | `index:coordinate` |
| Adjudicate | `index:coordinate` |
| Package and asset bytes | `index:qa:read` (plus an accepted, unrevoked assignment) |
| Publish, withdraw | `index:publish` |

A contributor holding only `index:intake` can create and read their own case, read shared
events, respond, appeal, escalate and repair. `package`/`asset` also read case metadata, so
that token needs a case-read scope too.

### Revoke and recover

- A coordinator revokes a reviewer with `index qa revoke ASSIGNMENT` (`index_qa_assignment_revoke`).
  The backend refuses later reads for that assignment, including package bytes; the SDK
  cannot override that.
- Revoking an OAuth grant or API key is done where it was issued (for example your agent's
  connection settings). The SDK holds no refresh token; after revocation every
  call is refused (exit 3) until you reconnect. Recovery is to reconnect with a
  token holding the needed scopes; work already submitted lives on the backend, not in the SDK.
- After an uncertain response, re-read state (`index contribution status`, `index qa case`)
  before deciding; retry only with the same idempotency key.

### Troubleshooting

| Symptom | Meaning and remedy |
| --- | --- |
| CLI exit 3 | Not authenticated. Check `SYNTH_API_KEY`, `--backend-url`, and whether the credential was revoked. |
| CLI exit 4, `insufficient_scope` | The message lists needed scopes (any-of) and the token's granted scopes. Reconnect and approve the missing scope. If the server sent no granted list it says `unknown`; it never guesses. |
| CLI exit 4, "not reported as a scope failure" | Scope class was not the reported cause. Check role, accepted/unexpired/unrevoked assignment, ownership and organization. |
| Exit 5 | Stale version or fence, or conflict. Re-read the case, use the new `expected_version`; nothing was merged. |
| Exit 6 | Rate limit or unavailable. Retry with the SAME idempotency key. |
| Exit 7 | Local or contract validation refusal; nothing was sent, or the response did not match the contract (for example a pre-v0.2 backend). |
| QA route unavailable | The backend has QA disabled. The SDK does not simulate it. |
| Local MCP tool missing | Mutating tools need `SYNTH_INDEX_MCP_WRITE_ENABLED=true` and an explicit key. |

MCP tool failures carry `insufficient_scope`, `required_scopes_any_of`, `granted_scopes`
(null when unreported) and a `hint` in the error data.

## Errors

### Public route

Every public-route failure is a `PublicSearchError` subclass with `.status` and
`.code`; none of them charges anything.

| Exception | HTTP / code | Meaning and remedy |
| --- | --- | --- |
| `PublicSearchRateLimitedError` | 429 `index_public_rate_limited` | A per-caller or platform-wide limit was hit. `.scope` names it (`peer_minute`, `peer_day`, `global_minute`, `global_day`); wait `.retry_after_s` (from `Retry-After`). |
| `PublicSearchBudgetExhaustedError` | 503 `index_public_budget_exhausted` | The free service budget is spent. `.scope` is `PublicSearchBudgetScope.DAILY_CENTS` or `DEEP_CONCURRENCY`; retry after `.retry_after_s` when the backend sends `Retry-After`. No result, no charge. |
| `PublicSearchRateStoreUnavailableError`, `PublicSearchMonitorUnavailableError` | 503 `index_rate_store_unavailable`, `monitor_unavailable` | The route failed closed; retry later. |
| `PublicSearchDisabledError` | 404 `index_public_search_disabled` | The backend has the public route turned off (or the mode is not offered). Use keyed Search or another deployment. |
| `PublicSearchAuthenticatedError` | 409 `index_public_search_authenticated` | Credential conflict: the public route is anonymous-only and a key or session was sent. Use `PublicIndexClient` (or CLI `--public`) with no key, or the paid keyed `search(...)`. |
| `PublicSearchRequestTooLargeError` | 413 `index_request_too_large` | Shorten the query. |
| `PublicSearchNotFoundError` | 404 `index_search_not_found` | Unknown Deep search id or wrong token. |
| `PublicSearchFailedError`, `PublicSearchCancelledError`, `PublicSearchWaitTimeoutError` | terminal state / local wait | Deep ended without delivery, or the local wait ended (the handle stays valid). |

The CLI prints the same messages and exits non-zero. Its own route conflicts
(`--public` with `--keyed`, `--api-key` or a keyed-only option; `--keyed` without a
key; no key and no route) are usage errors (exit 2) raised before any request.

### Keyed route

Transport raises typed `SynthError` subclasses: `RateLimitedError` (with
`retry_after_seconds` from `Retry-After`), `PaymentRequiredError` (private cap or
wallet), `AuthorizationError` (including uninvited private scope), `ConflictError`,
`TransientServiceError`. `index_error_code(error)` returns the stable
`IndexErrorCode`. The MCP tool raises `WalletConsentRequiredError`
(`index_wallet_consent_required`) before any paid request when the organization
has not turned on wallet payments. An unavailable service is never reported as an
empty result.

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
`TerminalRevisionError` instead of being submitted again. If storage refuses a signed
target because it expired, intake prepares again once and transfers only what is
still missing.

State (`synth.index.research-intake-state.v2`) names the backend, organization
and account it was allocated under, and is refused against any other — an
allocated draft and its idempotency keys mean nothing there. A `.lock` sidecar
holds the state file for one process at a time and names its holder, so a stale
lock is cleared deliberately. Corrupt or foreign state is reported with what to
do about it; it is never silently discarded.

Frozen-input release contracts live in `synth_ai.sdk.index.research`. The
authenticated `client.index.contributions.research` resource exposes
`allocate_archive`, `bind`, `consent`, `attest`, `revoke`, `release` and `archive`.
Archive allocation returns the versioned `ResearchArchiveAllocation` receipt with
only its collection and authorized snapshot scope; internal storage namespace IDs
remain server-side. The methods use the backend's exact revision routes and closed models. Binding
and consent retries reuse the same exact manifest and disclosure; the backend
rejects conflicting content. Consent is an explicit author action. Neither
binding nor attestation automatically consents or publishes.

`PublicIndexClient(...).contributions.release_research.retrieve(reference)` reads
only the safe released-output disclosure and scope/outcome summary. Historical
unbound releases return `None`. Private bindings, sessions and receipt evidence
are rejected in that response. The public resource has no archive operation.
Authenticated `research.archive(reference)` requires a current explicit archive
grant; ordinary Search does not acquire that permission from this SDK method.

This SDK surface does not yet implement native capture or isolated recipe
execution. The backend's maintained `research_bundle` commands own the current
offline build/reconstruction implementation; installing this client alone does
not qualify reproducibility or the local FAST/DEEP and browser gates.


Private snapshot transfers use `research.prepare_archive_upload(contribution_id,
snapshot_id, spec)` and `research.finalize_archive_upload(contribution_id,
snapshot_id, publication_id, collection_id=...)`. Prepare accepts only revision 1
with schema `synth.research.snapshot.v1`, and validates exact publication,
collection and declared object identity before exposing any transfer target.
Already stored objects can omit targets. Upload bytes through the existing
contribution upload helper. Finalize must return the same committed identity.
These operations allocate no release visibility or consent. Lost/expired fences
return `research_lease_lost`; retry preparation explicitly with the same identity.

`synth-ai index research capture-codex-rollout-prefix` freezes a selected native
JSONL prefix with `--native-input`, `--thread-id`, `--captured-at`, `--cutoff-at`
and `--out`. The input must end at the chosen cutoff and on a complete JSONL
record. Admission verifies native session aliases and timestamp order and retains
exact source bytes as `application/x-ndjson`. Capture is private and create-only;
a retry must match every byte. The declared partial coverage does not establish
inherited session history or external provider state.
