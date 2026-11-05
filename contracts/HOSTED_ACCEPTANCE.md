# Hosted MCP acceptance

`hosted-mcp-tools.json` is retained historical evidence for backend
`98762b3cb784fcf488621ea293c82f98f8c0380b`; it is not current deployment evidence.
The current generated source catalog is `mcp-tools.json`. Unit conformance checks
compare that catalog with the generated SDK and operation schemas.

The new source adds file upload and label tools. A fresh HTTP/Claude Code/Cursor
rehearsal must capture the installed tool list and revocation result before
hosted acceptance can be reported for the new pin. Do not relabel the historical
capture with the current commit or infer deployment from generated parity.
