# Coding-agent setup prompt for Synth Index

Copy the text below into a coding-agent task after you have chosen the Synth
backend environment and an authorized way to supply its API key. This prompt
configures an API/MCP integration; it does not ask the agent to build an Index
product UI or to run a paid search during setup.

> Set up Synth Index for this coding project through the `synth-ai` Python SDK
> and its `synth-ai-index-mcp` stdio server. Index usage is API/MCP-only. Do
> not add a search page, browser proxy, or customer-facing product workflow.
>
> First inspect the project's existing Python environment and coding-agent MCP
> configuration. Use the supported `synth-ai` package and verify that the
> `synth-ai-index-mcp` executable is available. Configure one MCP server with
> `SYNTH_INDEX_MCP_WRITE_ENABLED=false` and an
> explicit `SYNTH_BACKEND_URL` for the environment I named. Supply
> `SYNTH_API_KEY` only through this project's already-authorized, non-committed
> secret-injection mechanism. Never print, commit, paste into chat, or place the
> key in tool arguments. Do not use macOS Keychain or another credential store
> without my explicit authorization for that operation. If the backend URL or
> authorized key source is missing, report exactly what is missing and stop.
>
> Verify that the server starts and advertises `index_search` (the public
> route, available with or without a key) and, with the key, `index_private_search`
> and `index_search_create`. Without a key it must not advertise private or
> durable-search lifecycle tools. Tool discovery must not execute a search or
> incur a charge.
>
> Public `index_search` (fast or deep) is priced, rate-limited and governed by the
> privacy terms the backend publishes in capabilities; the tool returns them under
> `terms` with every result. Never quote a price you did not read from there.
> For a private or wallet-funded search I explicitly request, first state the
> aggregate maximum charge and use a fresh idempotency key. Wallet funding
> requires `search.billing.allow_wallet=true` and a `max_charge_cents` ceiling
> at or above `capabilities.private_search.price_cents_per_search`; do not infer
> my wallet consent from a query. DEEP also needs a mode grant. If a response is
> uncertain, retry the same logical request with the **same** idempotency key,
> not a new charge attempt. Preserve returned Contribution and revision IDs when
> reporting evidence.
> Do not enable Contribution writes unless I separately ask for them.
>
> Report the package version, backend environment name (not a secret), MCP
> server configuration with secret values redacted, advertised Index tool names,
> and any blocker. Do not claim live search works unless a separately
> authorized search actually completes and yields a receipt.

The server entry point and environment flags are documented in
[`README.md`](README.md). The typed billing object is
`SearchBillingConstraints` in [`search.py`](search.py); the backend remains the
authority for grants, balances, prices, and receipts.
