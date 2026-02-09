# Project Notes

- PlanetScale now supports PostgreSQL. Treat PlanetScale as Postgres for schema, SQL, and migrations.

## Coding Style

Follow the **Synth Style** guide: `specifications/tanha/references/tigerstyle.md` (sibling repo).

- **Docstrings link to specs.** Use `/// See: specifications/tanha/...` (Rust) or `# See: specifications/tanha/...` (Python) on non-trivial public functions/modules.
- **All errors handled.** Rust: `thiserror`, never `unwrap()`. Python: never swallow exceptions.
- **Naming.** No abbreviations. Units last: `timeout_ms`, `retry_count_max`.
- **Comments say why.** Code says what. Comments say why. Specs say the full story.

## CRITICAL: Running the Backend Locally

**NEVER attempt to start the backend yourself.** Too many interdependent services and env vars.

**Always use `local_dev.sh` with Colima (Docker) for infrastructure:**

```bash
cd ../backend
colima start                # if not already running
./local_dev.sh up           # start infra (Postgres :65432, Redis :6379, MinIO :9000, HelixDB :6969)
eval $(./local_dev.sh env)  # export DATABASE_URL, REDIS_URL, etc.
```

Then start services on host (separate terminals):
```bash
# Rust backend on port 8080 (GEPA engine, interceptor, graph service)
cd ../rust_backend && PORT=8080 cargo run --release

# Python backend on port 8000 (API gateway)
cd ../backend && source .env.dev && uvicorn app.routes.main:app --host 0.0.0.0 --port 8000
```

**Rust backend MUST be on port 8080** (`RUST_BACKEND_PUBLIC_URL=http://localhost:8080` in `.env.dev`).

**For GEPA/eval jobs**, both backends must be healthy before running `run_gepa_*.py` or `run_eval.py` scripts.

## CRITICAL: Auth Basics (LocalAPI + SynthTunnel)

When using the local stack with a tunneled task app, there are **three different keys**. Do not mix them.

### 1) Synth API key (backend auth)
- **Use:** Authenticate calls to the Synth backend (Python + Rust).
- **Env:** `SYNTH_API_KEY` (SDK default) or `SYNTH_BACKEND_API_KEY` (explicit override in some scripts).
- **Header:** `Authorization: Bearer <SYNTH_API_KEY>`.
- **Applies to:** `SYNTH_BACKEND_URL` (typically `http://127.0.0.1:8080` in local dev).
- **Do NOT:** Send a tunnel worker token to the backend. It will 401.

### 2) Environment API key (task app auth)
- **Use:** Authenticate calls from the backend/tunnel into your local task app.
- **Env:** `ENVIRONMENT_API_KEY` (minted by `ensure_localapi_auth()`).
- **Header:** `x-api-key: <ENVIRONMENT_API_KEY>` (task app expects this).
- **Applies to:** Your task app’s `/health`, `/info`, `/rollout`, etc.

### 3) SynthTunnel worker token (tunnel auth)
- **Use:** Authenticate SynthTunnel relay → your local task app.
- **Value:** `tunnel.worker_token` from `TunneledLocalAPI.create(...)`.
- **Use in jobs:** `task_app_worker_token`.
- **Do NOT:** Use this as a backend API key. It is **only** for tunnel relay auth.

### Local stack recipe (correct auth wiring)
1) **Backend URL**: `SYNTH_BACKEND_URL=http://127.0.0.1:8080`
2) **Backend auth**: `SYNTH_API_KEY=sk_*` (valid key in local DB)
3) **Task app**: run locally with `ENVIRONMENT_API_KEY` set or auto-minted
4) **Tunnel**: create a SynthTunnel for the task app
5) **Job config**: set `task_app_url` to the tunnel URL and `task_app_worker_token` to the worker token

If you see `Invalid API key` on `/api/jobs/*`, you're sending the wrong key to the backend.
If you see `SYNTH_TUNNEL_ERROR: Invalid worker token`, you're sending the wrong token to the tunnel.
