# Managed Research Plugin

This `synth-ai` plugin is the maintained local plugin surface for the
canonical `managed-research` MCP server.

Preferred hosted install:

- `codex mcp add managed-research --url https://api.usesynth.ai/mcp`
- `claude mcp add --transport http managed-research https://api.usesynth.ai/mcp`

Local stdio fallback:

- this plugin delegates to the sibling `managed-research` checkout
- the maintained stdio entrypoint is `managed-research-mcp`
- the maintained package install path is `uv tool install managed-research`

Canonical control-plane flow:

1. create a runnable project
2. inspect or prepare project setup
3. run launch preflight
4. trigger the run
5. inspect semantic progress
