# Project Notes

- PlanetScale now supports PostgreSQL. Treat PlanetScale as Postgres for schema, SQL, and migrations.

## Coding Style

Follow the **Synth Style** guide: `specifications/tanha/references/synthstyle.md` (sibling repo).

- **Docstrings link to specs.** Use `/// See: specifications/tanha/...` (Rust) or `# See: specifications/tanha/...` (Python) on non-trivial public functions/modules.
- **All errors handled.** Rust: `thiserror`, never `unwrap()`. Python: never swallow exceptions.
- **Naming.** No abbreviations. Units last: `timeout_ms`, `retry_count_max`.
- **Comments say why.** Code says what. Comments say why. Specs say the full story.

## CRITICAL: Running The Backend Locally

**Do not hand-roll the local stack.** Use the Synth-Bazel runtime manager.

```bash
cd ../synth-bazel
./scripts/runtime.py up
./scripts/runtime.py claim-slot alpha
./scripts/runtime.py build rust --profile dev-fast
./scripts/runtime.py services up alpha --profile local-only
eval "$(./scripts/runtime.py env alpha --exports)"
```

This starts shared infra and a managed local slot with the Python backend, Rust backend,
and worker on deterministic slot ports.

For GEPA/eval jobs, source local runtime env only from:

```bash
./scripts/runtime.py env <nickname> --exports
```

## CRITICAL: Auth Basics (Container + SynthTunnel)

When using the local stack with a tunneled container, there are **three different keys**. Do not mix them.

### 1) Synth API key (backend auth)
- **Use:** Authenticate calls to the Synth backend (Python + Rust).
- **Env:** `SYNTH_API_KEY` (SDK default) or `SYNTH_BACKEND_API_KEY` (explicit override in some scripts).
- **Header:** `Authorization: Bearer <SYNTH_API_KEY>`.
- **Applies to:** `SYNTH_BACKEND_URL` (typically `http://127.0.0.1:8080` in local dev).
- **Do NOT:** Send a tunnel worker token to the backend. It will 401.

### 2) Environment API key (container auth)
- **Use:** Authenticate calls from the backend/tunnel into your local container.
- **Env:** `ENVIRONMENT_API_KEY` (minted by `ensure_container_auth()`).
- **Header:** `x-api-key: <ENVIRONMENT_API_KEY>` (container expects this).
- **Applies to:** Your container’s `/health`, `/info`, `/rollout`, etc.

### 3) SynthTunnel worker token (tunnel auth)
- **Use:** Authenticate SynthTunnel relay → your local container.
- **Value:** `tunnel.worker_token` from `TunneledContainer.create(...)`.
- **Use in jobs:** `container_worker_token`.
- **Do NOT:** Use this as a backend API key. It is **only** for tunnel relay auth.

### Local stack recipe (correct auth wiring)
1) **Backend URL**: `SYNTH_BACKEND_URL=http://127.0.0.1:8080`
2) **Backend auth**: `SYNTH_API_KEY=sk_*` (valid key in local DB)
3) **Container**: run locally with `ENVIRONMENT_API_KEY` set or auto-minted
4) **Tunnel**: create a SynthTunnel for the container
5) **Job config**: set `container_url` to the tunnel URL and `container_worker_token` to the worker token

If you see `Invalid API key` on `/api/jobs/*`, you're sending the wrong key to the backend.
If you see `SYNTH_TUNNEL_ERROR: Invalid worker token`, you're sending the wrong token to the tunnel.

## CRITICAL: Container Auth Payload Rule (NO EXCEPTIONS)

- `container_api_key` and `container_api_keys` are forbidden in client-submitted policy-optimization
  payloads (`config_body`, overrides, and nested `prompt_learning.*` fields).
- Backend is the source of truth for rollout auth and must resolve credentials from org storage.
- If you touch payload builders, request schemas, or job creation paths, preserve this rule and add
  tests that assert these fields are rejected/stripped.

## Incident Log Requirement

When you hit a Synth code bug or local dev setup issue, append a timestamped entry to `/Users/joshpurtell/Documents/Github/specifications/issues_log/YYYY-MM-DD.md` before finishing.

Use this one-line format:
`- [YYYY-MM-DD HH:MM:SS TZ] <repo/path> — <issue> — <impact> — <action/status>`
