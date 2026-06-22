# Production logging lookup

This note records the local commands for getting production error context from
Sentry, Railway, and Vercel without printing secrets.

## What to collect for an incident

Always start with stable identifiers:

- timestamp with timezone
- environment: production, staging, preview, or local slot
- service: frontend, backend `api`, `worker-prod`, `smr-runtime-prod`, etc.
- route or UI action
- HTTP status and response body, if available
- project id, run id, task id, actor id, deployment id, or request id
- Sentry issue/event id, if available
- Railway or Vercel log command used to prove the failure

For launch incidents, also record the finding in the relevant backend release
packet or bug ledger. Logs are evidence, not the incident record.

## Sentry

Sentry is wired in both the frontend and backend.

Frontend:

- Next.js uses `@sentry/nextjs`.
- Local frontend Sentry build/upload env lives in `../frontend/.env.local`.
- Vercel production has `NEXT_PUBLIC_SENTRY_DSN`, `SENTRY_ORG`,
  `SENTRY_PROJECT`, and `SENTRY_AUTH_TOKEN`.

Backend:

- Backend initializes `sentry-sdk[fastapi]` from `core/o11y/sentry.py`.
- Railway production has `SENTRY_DSN`, `SENTRY_TRACES_SAMPLE_RATE=0.1`, and
  `SENTRY_PROFILES_SAMPLE_RATE=0.0` on the main prod services checked locally:
  `api`, `worker-prod`, and `smr-runtime-prod`.

### Load local Sentry CLI env

Use the read token in `../synth-ai/.env` for local incident lookup:

- `SENTRY_READ_AUTH_TOKEN`
- Created from Safari on 2026-05-06 as `Codex local prod logging read
  2026-05-06`.
- Scopes: `event:read`, `org:read`, `project:read`.

Do not use `awk -F=` for Sentry tokens. Tokens may contain `=` and env values
may be quoted. Use dotenv parsing:

```bash
cd /Users/joshpurtell/Documents/GitHub
eval "$(node -e 'const fs=require("fs"); const dotenv=require("./frontend/node_modules/dotenv"); const front=dotenv.parse(fs.readFileSync("./frontend/.env.local")); const synth=dotenv.parse(fs.readFileSync("./synth-ai/.env")); const vals={SENTRY_AUTH_TOKEN:synth.SENTRY_READ_AUTH_TOKEN,SENTRY_ORG:front.SENTRY_ORG,SENTRY_PROJECT:front.SENTRY_PROJECT}; for (const [k,v] of Object.entries(vals)) console.log("export "+k+"="+JSON.stringify(v||""));')"
```

Use the frontend-local CLI:

```bash
/Users/joshpurtell/Documents/GitHub/frontend/node_modules/.bin/sentry-cli info
```

Expected local result as of 2026-05-06:

- server: `https://sentry.io`
- org: `synth-jk`
- project: `frontend`
- auth method: auth token
- user: `josh@usesynth.ai`
- visible token scopes: `event:read`, `org:read`, `project:read`

### Query issues

Frontend project:

```bash
/Users/joshpurtell/Documents/GitHub/frontend/node_modules/.bin/sentry-cli issues list \
  --org "$SENTRY_ORG" \
  --project "$SENTRY_PROJECT" \
  --status unresolved
```

Backend project:

```bash
/Users/joshpurtell/Documents/GitHub/frontend/node_modules/.bin/sentry-cli issues list \
  --org "$SENTRY_ORG" \
  --project python-fastapi \
  --status unresolved
```

Verified on 2026-05-06:

- `frontend` unresolved issues list succeeds.
- `python-fastapi` unresolved issues list succeeds.

Prefer Sentry for exception traces, affected releases, fingerprints, request
tags, user/org tags, and grouped frontend/backend application failures.

## Railway

Railway is the first place to check backend process logs, service crashes,
runtime failures, DB errors, and deployment/build failures.

List prod services and deployment status:

```bash
cd /Users/joshpurtell/Documents/GitHub/backend
railway service status --environment production --all
```

Current prod services observed locally on 2026-05-06:

- `api`
- `worker-prod`
- `smr-runtime-prod`
- `sublinear-prod`
- `rhodes-worker-prod`
- `horizons-private-prod`
- `smr-git-server`
- `victorialogs-prod`
- `Redis`

Fetch recent logs for a service:

```bash
railway logs --environment production --service api --since 30m --lines 200
railway logs --environment production --service worker-prod --since 30m --lines 200
railway logs --environment production --service smr-runtime-prod --since 30m --lines 200
```

Filter likely errors:

```bash
railway logs --environment production --service api --since 2h --lines 300 --filter "@level:error"
railway logs --environment production --service smr-runtime-prod --since 2h --lines 300 --filter "run_id OR project_id OR error"
```

Use JSON when collecting evidence:

```bash
railway logs --environment production --service api --since 1h --lines 200 --json
```

Build/deploy logs:

```bash
railway logs --environment production --service api --deployment --latest --lines 200
railway logs --environment production --service api --build --latest --lines 200
```

Capture the command you ran and the relevant deployment id in the incident note.
For SMR failures, also capture project id, run id, actor id, task id, and any
blocker/resource fact visible through the backend API.

## Vercel

Vercel is the first place to check frontend server route handlers, Next.js
runtime errors, API proxy failures, build failures, and missing frontend env.

List deployments:

```bash
cd /Users/joshpurtell/Documents/GitHub/frontend
vercel ls --scope team_9xAfX44V2eyt9pNougeTENYe
```

Check production env presence without printing values:

```bash
vercel env ls --scope team_9xAfX44V2eyt9pNougeTENYe | rg "SENTRY|OPEN_RESEARCH|NEXT_PUBLIC"
```

Stream runtime logs for the production URL:

```bash
vercel logs https://www.usesynth.ai --scope team_9xAfX44V2eyt9pNougeTENYe
```

Fetch JSON logs for a deployment id or URL:

```bash
vercel logs <deployment-id-or-url> --json --scope team_9xAfX44V2eyt9pNougeTENYe
```

Important Vercel CLI limitation:

- `vercel logs` streams from now for up to about five minutes.
- For historical frontend failures, use Sentry if the event was captured, or
  reproduce while `vercel logs` is streaming.

## Practical triage order

For frontend-visible failures:

1. Check Vercel logs while reproducing.
2. Check Sentry for grouped frontend/server exceptions.
3. If the frontend route calls backend, capture backend HTTP status/body.
4. Check Railway `api` logs for the same timestamp and identifiers.

For SMR run failures:

1. Capture project id and run id from the UI/API.
2. Check backend run state, blockers, and usage/progress endpoints if relevant.
3. Check Railway `api` logs for request failures.
4. Check Railway `smr-runtime-prod` and `worker-prod` logs for execution
   failures.
5. Check Sentry for grouped backend exceptions using tags such as run id, project
   id, route, operation, or release where available.

For deployment failures:

1. Check Railway deployment/build logs for backend services.
2. Check Vercel deployment/build logs for frontend.
3. Check Sentry only after the service is running; it will not replace platform
   build logs.
