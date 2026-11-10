# Hosted MCP acceptance

The hosted `/mcp` endpoint serves the same generated catalog as the SDK:
`mcp-tools.json`, generated from Core's public contract (`x-synth-mcp-name`).
Unit conformance checks compare that catalog with the generated SDK and
operation schemas.

Generated parity is not deployment evidence. Hosted acceptance for a contract
pin needs a fresh HTTP/Claude Code/Cursor capture of the installed tool list and
a revocation result, recorded with the deployed Core identity. The earlier
capture from backend `98762b3` was removed on 2026-11-10: it predated the
public `pool` resource name (`rhodes.pool` there) and disagreed with the contract.
