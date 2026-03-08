# Synth Managed Research Codex Plugin

This plugin packages the existing `synth-ai` managed-research MCP server for Codex.

Included surfaces:

- `MCP`: `synth_managed_research` via `synth-ai-mcp-managed-research`
- `Skills`: setup, status, and run-monitoring workflows
- `App connector`: not bundled yet

## What it assumes

- `uv` is installed
- `SYNTH_API_KEY` is present in the environment
- `SYNTH_BACKEND_URL` is optional and only needed for non-default backends

## Current install paths

Preferred when your client supports app-server plugin install:

- marketplace: `synth-ai-local`
- plugin: `synth-managed-research`

Current Codex CLI fallback:

```toml
[mcp_servers.synth_managed_research]
command = "uv"
args = ["--directory", "<repo-root>", "run", "--quiet", "synth-ai-mcp-managed-research"]
env_vars = ["SYNTH_API_KEY", "SYNTH_BACKEND_URL"]
startup_timeout_sec = 20
tool_timeout_sec = 180
```

For the skills, either:

- install the plugin through an app-server client once `plugin/install` is available in your flow, or
- copy the skill folders into a Codex skill root such as `~/.codex/skills`

## Local marketplace intent

This repo also publishes `.agents/plugins/marketplace.json` so app-server clients can resolve
`pluginName = "synth-managed-research"` from marketplace `synth-ai-local`.

## Follow-on work

- Add a hosted `streamable_http` variant for team installs
- Add an optional app connector once Synth exposes a durable Codex-facing app id
- Add a dedicated setup/status MCP tool instead of relying on CLI doctor output
