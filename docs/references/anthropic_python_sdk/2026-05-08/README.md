# Anthropic Python SDK managed-agents reference

Snapshot date: 2026-05-08

Source repository: https://github.com/anthropics/anthropic-sdk-python

Snapshot commit:

```text
04b468daf76e4b95a949cecb03e29f4a1374d3b5
```

Snapshot tag:

```text
v0.100.0
```

Purpose: local reference for adding an analogous Synth SDK client for the
Horizons Private Anthropic-compatible managed-agents surface. This is not
vendored runtime code. It is a dated source snapshot used to compare namespace
shape, method names, pagination, streaming, raw-response wrappers, and async
coverage.

The upstream SDK is MIT licensed. The upstream license is copied at
[LICENSE](LICENSE).

## Snapshot contents

Only the generated beta resource files relevant to managed agents were copied:

| Upstream namespace | Local source snapshot |
| --- | --- |
| `client.beta` | [source/resources/beta/beta.py](source/resources/beta/beta.py) |
| `client.beta.agents` | [source/resources/beta/agents/agents.py](source/resources/beta/agents/agents.py) |
| `client.beta.agents.versions` | [source/resources/beta/agents/versions.py](source/resources/beta/agents/versions.py) |
| `client.beta.environments` | [source/resources/beta/environments.py](source/resources/beta/environments.py) |
| `client.beta.sessions` | [source/resources/beta/sessions/sessions.py](source/resources/beta/sessions/sessions.py) |
| `client.beta.sessions.events` | [source/resources/beta/sessions/events.py](source/resources/beta/sessions/events.py) |
| `client.beta.sessions.resources` | [source/resources/beta/sessions/resources.py](source/resources/beta/sessions/resources.py) |
| `client.beta.sessions.threads` | [source/resources/beta/sessions/threads/threads.py](source/resources/beta/sessions/threads/threads.py) |
| `client.beta.sessions.threads.events` | [source/resources/beta/sessions/threads/events.py](source/resources/beta/sessions/threads/events.py) |
| `client.beta.files` | [source/resources/beta/files.py](source/resources/beta/files.py) |
| `client.beta.skills` | [source/resources/beta/skills/skills.py](source/resources/beta/skills/skills.py) |
| `client.beta.skills.versions` | [source/resources/beta/skills/versions.py](source/resources/beta/skills/versions.py) |
| `client.beta.vaults` | [source/resources/beta/vaults/vaults.py](source/resources/beta/vaults/vaults.py) |
| `client.beta.vaults.credentials` | [source/resources/beta/vaults/credentials.py](source/resources/beta/vaults/credentials.py) |
| `client.beta.memory_stores` | [source/resources/beta/memory_stores/memory_stores.py](source/resources/beta/memory_stores/memory_stores.py) |
| `client.beta.memory_stores.memories` | [source/resources/beta/memory_stores/memories.py](source/resources/beta/memory_stores/memories.py) |
| `client.beta.memory_stores.memory_versions` | [source/resources/beta/memory_stores/memory_versions.py](source/resources/beta/memory_stores/memory_versions.py) |
| `client.beta.webhooks` | [source/resources/beta/webhooks.py](source/resources/beta/webhooks.py) |

## Key SDK shape to mirror

The Anthropic Python SDK exposes managed agents through `client.beta`, with
matching async resources on `AsyncAnthropic`. Each resource also exposes
`with_raw_response` and `with_streaming_response` wrappers.

Important top-level properties:

```python
client.beta.agents
client.beta.environments
client.beta.sessions
client.beta.vaults
client.beta.memory_stores
client.beta.files
client.beta.skills
client.beta.webhooks
```

Important nested properties:

```python
client.beta.agents.versions
client.beta.sessions.events
client.beta.sessions.resources
client.beta.sessions.threads
client.beta.sessions.threads.events
client.beta.skills.versions
client.beta.vaults.credentials
client.beta.memory_stores.memories
client.beta.memory_stores.memory_versions
```

The SDK sets managed-agents beta headers automatically for these resource calls.
The Synth SDK should do the same for both hosted Synth BFF usage and direct
Horizons Private usage.

## Primary method matrix

| Namespace | Methods |
| --- | --- |
| `beta.agents` | `create`, `retrieve`, `update`, `list`, `archive`; nested `versions.list` |
| `beta.environments` | `create`, `retrieve`, `update`, `list`, `delete`, `archive` |
| `beta.sessions` | `create`, `retrieve`, `update`, `list`, `delete`, `archive`; nested `events`, `resources`, `threads` |
| `beta.sessions.events` | `list`, `send`, `stream` |
| `beta.sessions.resources` | `retrieve`, `update`, `list`, `delete` |
| `beta.sessions.threads` | `retrieve`, `list`, `archive`; nested `events` |
| `beta.sessions.threads.events` | `list`, `stream` |
| `beta.files` | `list`, `delete`, `download`, `retrieve_metadata`, `upload` |
| `beta.skills` | `create`, `retrieve`, `list`, `delete`; nested `versions` |
| `beta.skills.versions` | `create`, `retrieve`, `list`, `delete` |
| `beta.vaults` | `create`, `retrieve`, `update`, `list`, `delete`, `archive`; nested `credentials` |
| `beta.vaults.credentials` | `create`, `retrieve`, `update`, `list`, `delete`, `archive`, `mcp_oauth_validate` |
| `beta.memory_stores` | `create`, `retrieve`, `update`, `list`, `delete`, `archive`; nested `memories`, `memory_versions` |
| `beta.memory_stores.memories` | `create`, `retrieve`, `update`, `list`, `delete` |
| `beta.memory_stores.memory_versions` | `retrieve`, `list` |
| `beta.webhooks` | resource class exists in the generated SDK; route behavior is defined by the current OpenAPI snapshot |

## Current Synth SDK gap

`synth_ai.sdk.managed_agents_anthropic.ManagedAgentsAnthropicClient` is a useful
thin HTTP client, but it is not shaped like the Anthropic SDK. It has flat
methods such as `create_agent(...)`, `create_session(...)`, and
`post_session_events(...)`.

For parity, add an Anthropic-shaped client facade alongside the existing flat
client instead of breaking existing eval harnesses.
