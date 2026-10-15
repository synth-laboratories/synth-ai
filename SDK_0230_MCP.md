# Unified MCP implementation — 0.23 local candidate

Implemented on the unpublished 0.23 development branch. Installed SDK/MCP and
local backend evidence is recorded with each release cohort. Hosted deployment,
native five-lane acceptance and production qualification remain separate gates.

## Runtime and catalog

Install the optional `synth-ai[mcp]` extra. `synth-ai-mcp` (also
`python -m synth_ai.mcp`) serves stdio using the official MCP Python SDK.
The SDK dependency is `mcp>=1.30,<2`; uv.lock pins the local build.

- `--profile public`: explicit anonymous public Index search; ignores ambient API keys.
- `--profile read` (default): keyed reads. Requires `SYNTH_API_KEY`.
- `--profile action`: keyed reads, paid searches, run creation and cancellation.
- `--base-url` or `SYNTH_API_URL`: one API origin (default `https://api.usesynth.ai`). Legacy `SYNTH_API_BASE_URL` and `SYNTH_BACKEND_URL` are accepted with a value-free deprecation warning.

Profiles narrow discovery and invocation; they never grant backend permission.
Do not put credentials in command-line arguments, tool parameters or MCP settings
visible to models. Stdio stdout belongs to the protocol. API-key retrieval stays
in the existing explicit/environment mechanism; no Keychain access was added.

`synth_ai/mcp/catalog.py` owns one typed adapter catalog for both transports.
`python -m synth_ai.mcp.schema` generates `contracts/mcp-v1.json`; the wheel and
sdist carry this exact artifact. The catalog references SDK operation IDs, input
models, result models, scope, mutation and charging semantics. Backend adoption
must reconcile these provisional definitions with its authoritative contracts.
It is not a second implementation of product policy.

Tools cover current identity; public and keyed Index search; durable search
create/list/status/result/events/usage/items/cancel; exact content and mutation
recovery; Swarms project/preflight/create/list/status/configuration/result/tasks/
events/usage/items/cancel; output list/metadata/verified text; file list/metadata.
`auth_context` reports the current principal without credentials. The
`synth://reference/tools` resource describes only this connection's visible tools.

No arbitrary HTTP tool, shell, provider key, organization override, local path,
base64 image input, upload transfer URL, or bulk all-pages traversal is exposed.
Large binary downloads and local file uploads remain explicit SDK operations.
Index contribution/account writes and richer runtime controls remain outside
this first MCP adapter catalog, even where an SDK method already exists.

## Mutation and result behavior

Mutation tools take a required stable `idempotency_key`. The SDK derives the
payload digest and exact route identity from validated inputs. Reuse the same
key and exact intent for recovery; changing either is a backend conflict, not a
new silent attempt. `MutationOptions` also accepts a full retained identity;
full identity and key are mutually exclusive.

Results preserve structured content, typed metadata, citations, usage/cost/charge
and timing. Errors are structured `isError` results with safe codes, request IDs,
retry hints and retained identity/key when available. Raw validation values,
exception strings, tokens and HTTP bodies are not reflected. Calls have a
60-second adapter deadline and 1 MiB structured-result ceiling. Observer tools
return bounded pages, not long-running subscriptions. A local cancellation or
timeout never requests remote cancellation. A created handle remains useful
without an extra post-create status request that could mask a successful launch.

Read-only hints do not imply zero cost. The tool metadata records charging
semantics separately. Search/run creation and explicit cancellation are mutation
operations. File and search output text is untrusted data, not tool instructions.

## Hosted authentication integration

`create_http_app(backend, resource_url=..., issuer_url=..., enabled_tools=...)`
constructs one stateless Streamable HTTP resource server. It uses official MCP
bearer middleware, per-request auth context, protected-resource metadata and
resource-audience validation. DNS rebinding protection and request-body limits
are enabled. Resource and issuer must be explicit HTTPS URLs. Enabled tools are
an explicit deployment allowlist, intersected with verified grants on every call.

The injected `AuthorizationBackend` is an actual backend integration requirement:

1. `verify(credential, resource=...)` selects the credential verifier through a
   recognized registry/format, validates active API keys or OAuth issuer/audience,
   expiry/revocation, organization membership and grants, and returns a typed
   `VerifiedPrincipal`. Invalid credentials return None. Outages raise and produce
   503; they never fall through to another verifier or anonymous authority.
2. Its principal includes server-owned subject/org/client attribution, scopes,
   allowed tools, expiry and an opaque server-only authorization handle. OAuth
   issuer and exact resource are checked again in the adapter. The incoming
   bearer is not retained in the tool execution context.
3. `invoke(principal, tool, arguments)` rechecks current rights, resource grants,
   entitlements, budgets and idempotency, then invokes shared backend services.
   It must retain actor/client attribution. It must not substitute an admin key
   or forward an MCP-audience browser token to the public API.

OAuth authorize/PKCE/consent/refresh belong to the configured authorization
server and client. This package does not invent authorization or registration
endpoints or mint broad API keys. Browser OAuth and API-key hosted journeys,
scope-upgrade challenges, revocation and supported-client compatibility remain
unqualified until integrated with the real backend and authorization server.
The concrete backend `AuthorizationBackend` implementation is still required;
passing a permissive verifier is not a supported production configuration.

Protocol sources used for the adapter design:
[official Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x),
[MCP authorization](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization),
[RFC 9728 protected-resource metadata](https://www.rfc-editor.org/rfc/rfc9728.html).

## Required verification, deliberately not executed

Run the installed SDK/MCP boundary suite, no-Any/source checks, stdio initialize/
discovery/invoke and HTTP protocol journeys after implementation review. Exercise
invalid/expired/revoked/cross-org credentials, API-key and OAuth parity, missing
scopes, unknown/disabled tools, schema errors, no secret reflection, idempotent
replay/conflict/uncertainty, pagination, cancellation and result limits. Follow
with native local-slot adoption and the live/cached/cloud release profiles.
The earlier 96-test result predates these changes and is not evidence for them.
