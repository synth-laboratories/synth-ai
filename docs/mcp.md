# MCP

Install `synth-ai[mcp]` and run `synth-ai-mcp` for stdio transport. Set the same
credentials and base URL as the Python client. The generated catalog matches the
core's hosted `/mcp` names, scopes and argument schemas. See [tools](mcp-tools.md).

For hosted OAuth, discover `/.well-known/oauth-protected-resource/mcp` and
`/.well-known/oauth-authorization-server`. `synth login` dynamically registers a
public client and uses PKCE S256 with state and issuer validation. Refresh tokens
are rotated and stored in a mode-0600 file. `SYNTH_AUTH_FILE` overrides that file.
`synth logout` revokes the grant before deleting the local record.

API-key management endpoints remain session-only: OAuth and bearer keys cannot
mint, update, rotate or revoke keys. Use the signed-in console for those actions.
