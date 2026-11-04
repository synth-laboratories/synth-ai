# MCP

The hosted MCP serves the Swarms, account and billing tools at `/mcp` and signs
in with OAuth 2.1 (authorization code with PKCE S256, dynamic client
registration). Clients discover the authorization server from
`/.well-known/oauth-protected-resource/mcp` and
`/.well-known/oauth-authorization-server`.

Tools: one per CLI verb, listed in [mcp-tools.md](mcp-tools.md).

## Claude Code

```sh
claude mcp add --transport http synth https://core-api-staging-staging-acef.up.railway.app/mcp
```

Run `/mcp` in Claude Code and choose `synth` to sign in.

## Cursor

`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "synth": {
      "url": "https://core-api-staging-staging-acef.up.railway.app/mcp"
    }
  }
}
```

## Local server

`pip install 'synth-ai[mcp]'`, then run `synth-ai-mcp --profile action` with
`SYNTH_API_KEY` set (`--profile read` lists only read tools).

```json
{
  "mcpServers": {
    "synth": {
      "command": "synth-ai-mcp",
      "args": ["--profile", "action"],
      "env": {"SYNTH_API_BASE_URL": "https://core-api-staging-staging-acef.up.railway.app"}
    }
  }
}
```
