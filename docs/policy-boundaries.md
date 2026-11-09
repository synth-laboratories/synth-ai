# Public contract and retry policy

Core owns the pinned public OpenAPI, scope registry and per-operation retry eligibility. `scripts/gen_contract.py` generates models, operation metadata, MCP schemas, OAuth login scopes and version metadata. Regenerate and run `--check` after changing the contract or package version. `SOURCE.toml` binds to the exact Core commit and content hash; this candidate preserves the 8192 UTF-8-byte message limit.

`x-synth-max-bytes` produces a Pydantic UTF-8-byte validator. The CLI reads the same generated message limit. `x-synth-message-text` rejects Unicode-blank messages and NUL. Shared fixtures in `contracts/fixtures/message-limits.json` match Core's fixture source and cover ASCII, emoji and two-byte characters at both sides of the limit.

`x-synth-idempotency-key` means the operation uses a stable key header, not that all failures can be replayed. `x-synth-retry` separately declares `safe_read`, `deduplicated` or `never`. Verified reads and public POST operations declared `x-synth-idempotency: durable_key` retry 429/503 responses automatically. Core audits these through its durable tenant/subject/operation/body-bound reserve-and-replay middleware. Transport failures or explicit unknown outcomes return immediately with request/key identity retained. PATCH/DELETE and protocol mutations never automatically replay. The contract records each audit and the existing 24-hour completed-response retention window.

`RetryPolicy` bounds attempts, total monotonic deadline, exponential backoff and additive jitter. Retry-After is a minimum wait and is never shortened to fit a budget; if it cannot fit, the last response is returned as a typed API error. Existing `timeout` and `retries` arguments remain supported and derive the default policy. Cancellation propagates. User-Agent and exported version are generated from project metadata.

Run `uv run pytest -q`, `uv run ty check`, and both generator checks. These are offline checks; no hosted acceptance or release is implied. Package acceptance checks use explicit errors and also fail under optimized Python.

Launch contract 2 adds hard routing selectors and a mandatory positive exact
budget. SDK 0.28.0 is a breaking candidate for clients using removed task-count
or provider/model fields. Core retains the authoritative catalog and admission
policy. The SDK validates structure; Core additionally validates catalog
membership, feasibility, every actor and fixed-profile compatibility before
funding. `RunGuidance` uses the same UTF-8 byte and NUL checks as Core.

Malformed successful mutation responses raise `ResponseContractError` with
`outcome="unknown"`, request identity and the original idempotency key. This is
not evidence of rejection and must not trigger a new mutation identity.

Launch contract 3 / SDK 0.29.0 replaces `program`, `method` and
`profile_overrides` with a named `profile` and optional `research_task`.
`GET /v1/profiles` is scoped read-only discovery; its availability is specific
to caller and task and must not be shared across credentials. It includes no
approval documents or secrets. Requests cannot author a profile configuration.
The ordinary profile is Swarms v2; research profiles retain the existing
approval checks and immutable parameters. Accepted runs persist their resolved
profile pins with the existing program selection.
