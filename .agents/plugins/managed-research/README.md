# Managed Research Plugin

This `synth-ai` plugin is the maintained local plugin surface for the
canonical Research MCP server shipped in `synth-ai[research]`.

Preferred hosted install:

- `codex mcp add synth-managed-research --url https://api.usesynth.ai/mcp`
- `claude mcp add --transport http synth-managed-research https://api.usesynth.ai/mcp`

Local stdio fallback:

- this plugin delegates to the sibling `synth-ai` checkout
- the maintained stdio entrypoint is `synth-ai-managed-research-mcp`
- the maintained package install path is `pip install "synth-ai[research]"`

Canonical control-plane flow:

1. create a runnable project
2. inspect or prepare project setup
3. run launch preflight
4. trigger the run
5. inspect semantic progress
