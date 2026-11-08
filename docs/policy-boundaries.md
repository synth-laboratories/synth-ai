# Public contract and retry policy

Core owns the pinned public OpenAPI, scope registry and per-operation retry eligibility. `scripts/gen_contract.py` generates models, operation metadata, MCP schemas, OAuth login scopes and version metadata. Regenerate and run `--check` after changing the contract or package version. `SOURCE.toml` binds to the exact Core commit and content hash; this candidate preserves the 8192 UTF-8-byte message limit.

`x-synth-max-bytes` produces a Pydantic UTF-8-byte validator. The CLI reads the same generated message limit. Shared fixtures in `contracts/fixtures/message-limits.json` match Core's fixture source and cover ASCII, emoji and two-byte characters at both sides of the limit.

`x-synth-idempotency-key` means the operation uses a stable key header, not that all failures can be replayed. `x-synth-retry` separately declares `safe_read`, `deduplicated` or `never`. Only verified reads, run creation and checkout creation retry 429/503 responses automatically. Transport failures or explicit unknown outcomes return immediately with request/key identity retained. Other mutations never automatically replay.

`RetryPolicy` bounds attempts, total monotonic deadline, exponential backoff and additive jitter. Retry-After is a minimum wait and is never shortened to fit a budget; if it cannot fit, the last response is returned as a typed API error. Existing `timeout` and `retries` arguments remain supported and derive the default policy. Cancellation propagates. User-Agent and exported version are generated from project metadata.

Run `uv run pytest -q`, `uv run mypy synth_ai`, and both generator checks. These are offline checks; no hosted acceptance or release is implied. Package acceptance checks use explicit errors and also fail under optimized Python.
