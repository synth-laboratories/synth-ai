# Synth SDK 0.23 development branch

Local SDK-first Index implementation. Package metadata is `0.23.0.dev2026101309`;
this source candidate is unpublished and is not a release qualification.

The new client targets the closed `index.v1` contract on `/api/v1`; it refuses
legacy responses and does not fall back. Backend adoption and native/cached/live
acceptance are pending. This worktree also implements the first Swarms core,
output/file delivery and unified MCP adapter. The initial caller model override
accepts only `gpt-5.6-luna`, with `openai` as the supported provider override.
Omitted overrides preserve backend project defaults; resolved metadata is not
an override allowlist. Hosted MCP requires the backend authorization/service
adapter; its native OAuth/API-key journeys remain unqualified.

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

The retired source remains on `legacy/sdk-pre-0230-20261013`; no broad compatibility
aliases, old CLI or split MCP are shipped.

- [Implementation and contract choices](SDK_0230_IMPLEMENTATION.md)
- [Backend delta map](SDK_0230_DESIGN.md)
- [Testing expectations](SDK_0230_TESTING.md)
- [Unified MCP implementation and auth integration](SDK_0230_MCP.md)

Tests live in the dedicated testing repository and install the selected wheel
in an owned isolated environment. SDK unit/contract passes are not backend or
end-to-end qualification. No backend services/data/deployment are changed here.

## CLI authentication candidate

`synth-ai login` uses browser PKCE and a loopback callback. `--callback-port` selects a registered alternative when the default port is occupied; `--no-browser` writes the authorization URL to the private login file and prints only its path. `synth-ai auth status --json` verifies the current credential against the selected Synth API. `synth-ai logout` revokes stored OAuth credentials and clears local storage.

For automation, pass a Synth key on standard input to `synth-ai login --with-token`. `SYNTH_API_KEY` takes precedence over saved credentials. `SYNTH_API_URL` selects the API, with the legacy endpoint variable retained during migration. OAuth login requires a registered public application configured through `SYNTH_OAUTH_CLIENT_ID`, `SYNTH_OAUTH_ISSUER` and `SYNTH_OAUTH_RESOURCE`. No confidential client secret belongs in the CLI.

Credentials use the optional keyring backend when installed; `SYNTH_AUTH_STORAGE=file` selects private atomic file storage under `SYNTH_CONFIG_DIR` (default `~/.synth`). Test runs explicitly select file storage. Never publish credential files or the private login file.

`--device-auth` is currently gated in the verified Clerk development instance: device consent returns a token without the Synth audience, and configured RFC 8693 exchange returns `unsupported_grant_type`. The CLI returns `device_resource_binding_unavailable` before consent when discovery lacks that operation. A development-only Clerk support request is pending. Browser login and headless keys remain independent; successful protocol tests do not qualify a real device journey.

For an explicitly selected local backend candidate, pass `--candidate-manifest-sha256` to `auth status` using the retained candidate manifest. The SDK still rejects candidate operations without a pin and rejects candidate pins against production. Authentication does not waive release qualification.

Protocol fingerprints come from canonical JSON snapshots packaged in the wheel, rather than the installed Pydantic version’s schema renderer. A model or operation change must update the canonical exports and packaged snapshots together. The current snapshots match the backend protocol generated with Pydantic 2.12.5 and remain identical when the SDK runs with 2.14.1.
