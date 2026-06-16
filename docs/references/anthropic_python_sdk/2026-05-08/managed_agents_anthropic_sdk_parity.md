# Managed Agents Anthropic SDK parity plan

Date: 2026-05-08

Goal: make `synth-ai` provide an Anthropic-shaped Python client for the Synth
Managed Agents surface, so user code can be moved between Anthropic Managed
Agents and Horizons Private/Synth by changing the import/base URL, while still
using Synth auth and unsupported-feature errors where the backend does not yet
implement full parity.

Reference snapshot:

- `docs/references/anthropic_python_sdk/2026-05-08/`
- upstream `anthropics/anthropic-sdk-python` commit
  `04b468daf76e4b95a949cecb03e29f4a1374d3b5`, tag `v0.100.0`

## Target user experience

Anthropic-shaped direct Horizons Private usage:

```python
from synth_ai import SynthManagedAgents

client = SynthManagedAgents.from_horizons_private(
    base_url="http://127.0.0.1:8182",
)

environment = client.beta.environments.create(name="legal-review")
agent = client.beta.agents.create(
    name="reviewer",
    model="gpt-5.4-mini:codex",
    tools=[{"type": "agent_toolset_20260401"}],
)
session = client.beta.sessions.create(
    agent=agent["id"],
    environment_id=environment["id"],
)
client.beta.sessions.events.send(
    session["id"],
    events=[
        {
            "type": "user.message",
            "content": [{"type": "text", "text": "Write the memo."}],
        }
    ],
)
for event in client.beta.sessions.events.stream(session["id"]):
    print(event)
```

Synth-hosted BFF usage:

```python
from synth_ai import SynthClient

client = SynthClient()
agent = client.managed_agents_anthropic.beta.agents.create(
    name="reviewer",
    model="gpt-5.4-mini:codex",
)
```

Compatibility principle:

- Preserve the current flat `client.managed_agents` SDK for existing evals.
- Add an Anthropic-shaped facade as the preferred SDK parity surface.
- Use the same method names as Anthropic where possible:
  `create`, `retrieve`, `update`, `list`, `delete`, `archive`, `send`,
  `stream`, `upload`, `download`, `retrieve_metadata`.
- Return dict-like objects at first. Typed response models can come later.

## Namespaces to implement

| Synth facade namespace | Backing flat method or route | Backend status |
| --- | --- | --- |
| `beta.agents.create` | `create_agent` | supported |
| `beta.agents.retrieve` | `get_agent` | supported |
| `beta.agents.update` | `update_agent` | supported |
| `beta.agents.list` | `list_agents` | supported |
| `beta.agents.archive` | `archive_agent` | supported |
| `beta.agents.versions.list` | raw request to agent versions route or unsupported wrapper | partial/backend dependent |
| `beta.environments.create` | `create_environment` | supported |
| `beta.environments.retrieve` | `get_environment` | supported |
| `beta.environments.update` | `update_environment` | supported |
| `beta.environments.list` | `list_environments` | supported |
| `beta.environments.delete` | raw delete or archive fallback if delete absent | partial/backend dependent |
| `beta.environments.archive` | `archive_environment` | supported |
| `beta.sessions.create` | `create_session` | supported |
| `beta.sessions.retrieve` | `get_session` | supported |
| `beta.sessions.update` | `update_session` | supported |
| `beta.sessions.list` | `list_sessions` | supported |
| `beta.sessions.delete` | raw delete or archive fallback if delete absent | partial/backend dependent |
| `beta.sessions.archive` | `archive_session` | supported |
| `beta.sessions.events.list` | `list_session_events` | supported |
| `beta.sessions.events.send` | `post_session_events` | supported |
| `beta.sessions.events.stream` | `stream_session_events` | supported |
| `beta.sessions.resources.retrieve` | `get_session_resource` | supported |
| `beta.sessions.resources.update` | `update_session_resource` | supported |
| `beta.sessions.resources.list` | `list_session_resources` | supported |
| `beta.sessions.resources.delete` | `delete_session_resource` | supported |
| `beta.sessions.threads.retrieve` | raw request to thread retrieve route | partial/backend dependent |
| `beta.sessions.threads.list` | `list_session_threads` | supported |
| `beta.sessions.threads.archive` | raw request to thread archive route | partial/backend dependent |
| `beta.sessions.threads.events.list` | `list_session_thread_events` | supported |
| `beta.sessions.threads.events.stream` | raw stream to thread events route | partial/backend dependent |
| `beta.files.upload` | `create_file` plus file-path/multipart convenience | supported with better ergonomics needed |
| `beta.files.list` | `list_files` | supported |
| `beta.files.retrieve_metadata` | `get_file` | supported |
| `beta.files.download` | `download_file_content` | supported |
| `beta.files.delete` | `delete_file` | supported |
| `beta.skills.*` | raw request returning backend 501 until implemented | unsupported, loud |
| `beta.vaults.*` | raw request returning backend 501 until implemented | unsupported, loud |
| `beta.memory_stores.*` | raw request returning backend 501 until implemented | unsupported, loud |
| `beta.webhooks.*` | raw request returning backend 501 until implemented | unsupported, loud |

## Implementation chunks

1. Client shell

   Add `SynthManagedAgents` and `AsyncSynthManagedAgents` classes that own a
   `ManagedAgentsAnthropicClient` transport and expose `beta`.

2. Resource namespaces

   Add small resource classes under `synth_ai/sdk/managed_agents/`:

   ```text
   managed_agents/
     __init__.py
     client.py
     transport.py
     beta.py
     resources/
       agents.py
       environments.py
       sessions.py
       files.py
       skills.py
       vaults.py
       memory_stores.py
       webhooks.py
   ```

   Keep each resource thin: translate Anthropic-style method names to the
   existing flat transport.

3. Pagination and streaming

   Add minimal auto-paging wrappers for `.list(...)` responses with `data` and
   `next_page`/cursor fields. Keep raw dict pages available by default until
   typed models are added.

   Preserve streaming as an iterator of event dicts first. Later we can add
   event model unions.

4. File ergonomics

   Add `files.upload(file=..., purpose=..., metadata=...)` that accepts:

   - path string / `Path`
   - bytes
   - file-like object

   Map to whatever the Synth/Horizons service currently accepts, and raise a
   clear error if the backend does not support true multipart yet.

5. Unsupported advanced resources

   Implement the resource methods even when the backend returns 501. That makes
   user code and examples importable against Synth, while preserving loud
   runtime truth for unsupported features.

6. Async parity

   Mirror every sync resource with async methods. It is acceptable initially to
   wrap sync transport with `asyncio.to_thread`, matching the current async flat
   client pattern.

7. Exports

   Export:

   ```python
   from synth_ai import SynthManagedAgents, AsyncSynthManagedAgents
   ```

   Also wire:

   ```python
   SynthClient(...).managed_agents_anthropic
   AsyncSynthClient(...).managed_agents_anthropic
   ```

   Keep `SynthClient(...).managed_agents` pointing at the existing flat client
   during migration.

8. Docs and examples

   Add a README example showing the same flow against:

   - Anthropic: `from anthropic import Anthropic`
   - Synth: `from synth_ai import SynthManagedAgents`

   The code should differ only in import/client construction and model/base URL.

## Non-goals for first parity pass

- Do not generate a full Stainless-style typed SDK yet.
- Do not implement backend support for memory, dreams, outcomes, vaults,
  webhooks, MCP, or custom tools in `synth-ai`; the SDK should expose the method
  shape and let the backend return loud unsupported-feature errors.
- Do not remove the flat `ManagedAgentsAnthropicClient`; evals currently use it.

## First acceptance checks

- `client.beta.agents.create(...)` creates the same agent as
  `client.managed_agents.create_agent(...)`.
- `client.beta.sessions.events.send(...)` and `.stream(...)` can run the
  OpenAIReview smoke.
- `client.beta.files.upload/download/retrieve_metadata/list/delete` can round
  trip a file.
- `client.beta.sessions.threads.events.list(...)` can inspect a Harvey LAB
  child-agent thread.
- `client.beta.memory_stores.create(...)` returns the backend's structured 501
  unsupported-feature error, not an SDK `AttributeError`.
