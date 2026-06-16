# Managed Agents SDK

Anthropic-shaped facade for the Synth Managed Agents surface.

Use this when porting code written against Anthropic Managed Agents. The goal is
that the application code keeps Anthropic-style namespaces such as
`client.beta.agents`, `client.beta.sessions.events`, and `client.beta.files`.

```python
from synth_ai import SynthManagedAgents

client = SynthManagedAgents.from_horizons_private(
    base_url="http://127.0.0.1:8182",
)

environment = client.beta.environments.create(name="review-env")
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
            "content": [{"type": "text", "text": "Review the workspace."}],
        }
    ],
)
for event in client.beta.sessions.events.stream(session["id"]):
    print(event["type"])
```

Existing eval harnesses may continue using the flat
`ManagedAgentsAnthropicClient`. New user-facing examples should prefer
`SynthManagedAgents`.

Unsupported advanced Anthropic surfaces, such as memory stores, vaults, skills,
custom tools, MCP connectors, dreams, outcomes, and webhooks, intentionally keep
their SDK namespace shape and defer to the backend for loud
`unsupported_feature` responses.
