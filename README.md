# Synth SDK 0.23 development branch

Local SDK-first Index implementation. The pinned 0.23 development wheel is an
engineering qualification candidate and must not be published as a release.

The new client targets the closed `index.v1` contract on `/api/v1`; it refuses
legacy responses and does not fall back. Backend adoption and native/cached/live
acceptance are pending. This worktree also implements the first Swarms core,
output/file delivery and unified MCP adapter. The initial caller model override
accepts `gpt-6-luna`, with `openai` as the supported provider override.
Omitted overrides preserve backend project defaults; resolved metadata is not
an override allowlist. Hosted MCP requires the backend authorization/service
adapter; its native OAuth/API-key journeys remain unqualified.

SDK clients and `synth-ai-mcp` use `SYNTH_API_KEY` for authentication and
`SYNTH_API_URL` for the API origin (default `https://api.usesynth.ai`). Explicit
`api_key` / `base_url` arguments, or the MCP `--base-url` flag, take precedence.
Anonymous clients do not read key variables.

Without an explicit or environment key, authenticated clients and stdio MCP
use the private credential created by `synth-ai login`. Login discovery is lazy;
each authenticated operation reloads the origin-bound credential and refreshes
it if needed. Logout therefore affects the next call in an existing client.

During migration, `SYNTH_USER_KEY` and `SYNTH_BACKEND_API_KEY` remain key aliases;
`SYNTH_BACKEND_URL` and `SYNTH_API_BASE_URL` remain URL aliases. Using an alias
without the canonical variable emits one warning per name per process, without
its value. Conflicting legacy values refuse configuration. A present but empty
canonical variable is invalid and never falls back to another identity.

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import PublicSearchOptions

with SynthClient.anonymous() as client:
    result = client.index.search("reproducible retrieval evidence", options=PublicSearchOptions())
    print(result.response)
```

This requires an index.v1 backend. Keyed calls require KeyedSearchOptions with
an exact Money charge ceiling. Durable search and mutation recovery are keyed.
Use typed content refs for current-rights retrieval; citations grant no access.
Synchronous clients own a cancellable HTTP pool; synchronous client injection is
not supported. Async clients use native async HTTP and may borrow an AsyncClient.
Local observer timeouts and close never cancel durable remote work.

For explicit HTTPX use, `synth_ai.transports.DeadlineHTTPTransport` supplies the
same owned, cancellable network implementation. Set a finite HTTPX read timeout;
it bounds the whole exchange, including headers and body. Close the HTTPX client
to release its pool and join its network thread. Arbitrary non-cooperative
synchronous transports cannot be preempted; synchronous client injection into
`SynthClient` is deliberately unsupported. Observer total, idle, reconnect and
event-count bounds remain independent of explicit remote cancellation.

The retired source remains on `legacy/sdk-pre-0230-20261013`; no broad compatibility
aliases, old CLI or split MCP are shipped.

- [Implementation and contract choices](SDK_0230_IMPLEMENTATION.md)
- [Backend delta map](SDK_0230_DESIGN.md)
- [Testing expectations](SDK_0230_TESTING.md)
- [Unified MCP implementation and auth integration](SDK_0230_MCP.md)

Tests live in the dedicated testing repository and install the selected wheel
in an owned isolated environment. SDK unit/contract passes are not backend or
end-to-end qualification. No backend services/data/deployment are changed here.

Per-request capacity is available through a caller-owned
`synth_ai.types.rate_limits.ResponseMetadata` passed as
`MutationOptions(response_metadata=capacity)` or `ReadOptions(response_metadata=capacity)`.
Use one instance per concurrent request. Typed errors and MCP error payloads also
retain bounded rate-limit values. These fields never change the result body.

## Qualified publication

Source pushes do not publish packages. `.github/workflows/publish-dev.yml`
keeps the registered Trusted Publisher filename and `pypi` environment, and
requires a manual release dispatch. Nightlies use an accepted `staging` source
and an exact `X.Y.Z.devN` version; stable publication uses `main` and `X.Y.Z`.
The selected source must already be on that branch. The publisher never builds
or restamps a distribution.

The dispatch selects a same-repository Actions artifact by immutable ID and
archive SHA256, plus the exact source SHA and version. Its successful producer
run must name that source and must not be a PR or fork run. The artifact contains:

- `qualification.json` with schema `synth.sdk.qualified-artifact.v1`,
  `status: "passed"`, `source_sha`, `version`, and `channel`.
- `distributions`: one wheel and optionally one sdist, each with `filename` and
  SHA256. These are the final-version bytes that were qualified.
- `checks`: `sdk-contracts`, `fresh-install`, `native-index-fast`,
  `native-index-deep`, `native-monitor`, `native-swarms`, and `staging`.
  Each carries `status: "passed"` (see deferral below), the same `source_sha`, the exact
  `distributions` filename-to-SHA256 mapping, `evidence_path` under `evidence/`,
  and that file's `evidence_sha256`. A check named in `.github/release-policy.json`
  for that exact version may instead carry `status: "deferred"` with a non-empty
  `reason` and a ticket id (`ticket`); its evidence file is the deferral record
  `{"check", "reason", "ticket"}` and must match. For 0.23.0 only `native-swarms`
  is deferrable. Every other check must be `passed`, and the publish receipt
  lists deferred checks.

Qualification must produce this artifact only after the named checks actually
pass. The verifier binds retained evidence and bytes; it does not infer test
success from filenames. Review the source, checks and wheel hashes in the
verification job before authorizing the protected `pypi` job. Existing registry
versions are accepted only when their complete filename/hash set matches;
different bytes refuse instead of being silently skipped. The final registry
readback and publication receipt are retained as workflow artifacts.

For qualification performed on the release host, retain the complete reviewed
ZIP as a same-repository release asset, then dispatch
`retain-qualified-release.yml` at its exact source commit. Supply the asset ID,
ZIP hash, channel and final version. The producer validates the same source,
distribution and evidence rules and uploads the unchanged files as an Actions
artifact. Use that artifact's immutable ID and archive digest in the protected
publisher. The producer does not run missing checks or build a replacement wheel.

`.github/v023-preserved-candidate.json` records the unchanged SDK1511 candidate
wheel and original source. That record is provenance, not publication approval
or a substitute for final native and staging qualification.


### Tunnel v2 control (development candidate)

`SynthClient(...).tunnels` and `AsyncSynthClient(...).tunnels` expose the canonical
`/api/v1/synthtunnel/v2` connector, route, lease and session APIs. Typed requests
are available from `synth_ai.sdk.tunnels`; checkpoint commands are in
`synth_ai.sdk.tunnels.session_operations`. The client owns only HTTP resources;
borrowed lease handles never shut down the shared connector.

This catalog protocol retains backend UUID/deadline authority and dedicated
command/session lookup. An uncertain mutation is observed through those lookup
methods before another effect is considered. Closing a client or cancelling
observation does not prove origin cancellation. Tokens remain distinct from
backend API credentials and are redacted outside explicit wire serialization.

The existing Index/Swarms capability envelopes and schema fingerprints are
unchanged. Their release/candidate manifest pins cannot qualify Tunnel catalog
v2; such pinned clients refuse control dispatch. Use a separate control client
with its explicit backend origin and API credential. No credential-bearing
redirect or alternate origin is followed. Installed-package protocol tests do
not establish deployed backend/Rivet acceptance or package publication.
